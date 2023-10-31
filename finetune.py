import logging
import os
from math import ceil
from typing import Optional, Tuple
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="intel_extension_for_pytorch"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io.image", lineno=13
)

import torch
import intel_extension_for_pytorch as ipex
from datasets import load_dataset
from datasets import Dataset
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.qlora import (
    get_peft_model,
    prepare_model_for_kbit_training as prepare_model,
)
import wandb
from fire import Fire
from peft import LoraConfig
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
wandb.init(project="LLM-FineTuning")


# TODO: Move these to a config file later
BASE_MODEL = "openlm-research/open_llama_3b"
MODEL_PATH = "./model"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def generate_prompt_book(text: str):
    return f"""You are an AI trained in the style of classic literature. 

### Text:
{text}

### Response:"""


class FineTuner:
    """A class to handle the fine-tuning of LLM models."""

    def __init__(self, base_model_id: str, model_path: str, device: torch.device):
        """
        Initialize the FineTuner with base model, model path, and device.

        Parameters:
            base_model (str): The pre-trained model to use for fine-tuning.
            model_path (str): Path to save the fine-tuned model.
            device (torch.device): Device to run the model on.
        """
        self.base_model_id = base_model_id
        self.model_path = model_path
        self.device = device

    def setup_models(self):
        """Setup download and save base models."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                load_in_low_bit="nf4",
                optimize_model=True,
                torch_dtype=torch.float16,
                modules_to_not_convert=["lm_head"],
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        except Exception as e:
            logging.error(f"Error in downloading models: {e}")

    def tokenize_batch(self, data_points, add_eos_token=True, cutoff_len=512) -> dict:
        """Tokenize a batch of text."""
        try:
            texts = data_points["text"]
            structured_prompts = [generate_prompt_book(text) for text in texts]
            results = self.tokenizer(
                structured_prompts,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if add_eos_token:
                for i, tokens in enumerate(results["input_ids"]):
                    if len(tokens) < cutoff_len:
                        tokens.append(self.tokenizer.eos_token_id)
                        results["attention_mask"][i].append(1)
            results["labels"] = [ids.copy() for ids in results["input_ids"]]
            return results
        except Exception as e:
            logging.error(
                f"Error in batch tokenization: {e}, Line: {e.__traceback__.tb_lineno}"
            )
            raise e

    def prepare_data(self, data, val_set_size=100) -> Dataset:
        """Prepare training and validation datasets."""
        try:
            train_val_split = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val_split["train"].map(
                lambda x: self.tokenize_batch(x), batched=True
            )
            val_data = train_val_split["test"].map(
                lambda x: self.tokenize_batch(x), batched=True
            )
            return train_data, val_data
        except Exception as e:
            logging.error(
                f"Error in preparing data: {e}, Line: {e.__traceback__.tb_lineno}"
            )
            raise e

    def train_model(self, train_data, val_data, training_args):
        """
        Fine-tune the model with the given training and validation data.

        Parameters:
            train_data (Dataset): Training dat        tokenizer = AutoTokenizer.from_pretrained(self.base_model)a.
            val_data (Optional[Dataset]): Validation data.
            training_args (TrainingArguments): Training configuration.
        """
        try:
            self.model = self.model.to(DEVICE)
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model(self.model)
            self.model = get_peft_model(self.model, LORA_CONFIG)
            trainer = Trainer(
                model=self.model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                data_collator=DataCollatorForSeq2Seq(
                    self.tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
            self.model.config.use_cache = False
            trainer.train()
            self.model.save_pretrained(self.model_path)
        except Exception as e:
            logging.error(f"Error in model training: {e}")

    def finetune(self, data_path, training_args):
        """
        Execute the fine-tuning pipeline.

        Parameters:
            data_path (str): Path to the data for fine-tuning.
            training_args (TrainingArguments): Training configuration.
        """
        self.setup_models()
        data = load_dataset("json", data_files=data_path)
        train_data, val_data = self.prepare_data(data)
        self.train_model(train_data, val_data, training_args)


if __name__ == "__main__":
    print(f"Finetuning on device: {ipex.xpu.get_device_name()}")
    try:
        finetuner = FineTuner(
            base_model_id=BASE_MODEL, model_path=MODEL_PATH, device=DEVICE
        )
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            save_steps=100,
            warmup_steps=20,
            # max_steps=200,
            learning_rate=2e-4,
            num_train_epochs=3,
            evaluation_strategy="steps",
            eval_steps=100,
            fp16=True,
            logging_steps=20,
            optim="adamw_hf",
            output_dir="./output",
            logging_dir="./logs",
            report_to="wandb",
        )
        data_path = "./book_data.json"  # TODO: Move this to a config file later
        finetuner.finetune(data_path, training_args)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
