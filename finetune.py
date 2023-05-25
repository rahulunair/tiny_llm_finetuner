import os
from math import ceil
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="intel_extension_for_pytorch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image", lineno=13)

import intel_extension_for_pytorch as ipex
import torch
from datasets import load_dataset
from fire import Fire
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from transformers import (DataCollatorForSeq2Seq, LlamaForCausalLM,
                          LlamaTokenizer, Trainer, TrainingArguments)


DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Finetuning on device: {ipex.xpu.get_device_name()}")

def get_device(self):
    if torch.xpu.is_available():
        return DEVICE
    else:
        return self.device


def place_model_on_device(self):
    self.model.to(self.args.device)


TrainingArguments.device = property(get_device, TrainingArguments.device.setter)
Trainer.place_model_on_device = place_model_on_device


class FineTuner:
    """
    A class to handle fine tuning of an LLM model.
    """

    MODEL_PATH = "./model"
    BASE_MODEL = "openlm-research/open_llama_3b_600bt_preview"

    def __init__(self, base_model=BASE_MODEL, model_path=MODEL_PATH, device=DEVICE):
        self.base_model = base_model
        self.model_path = model_path
        self.device = device

    @staticmethod
    def download_models():
        """
        Downloads and saves models.
        """
        model = LlamaForCausalLM.from_pretrained(FineTuner.BASE_MODEL)
        model.save_pretrained(FineTuner.MODEL_PATH)
        tokenizer = LlamaTokenizer.from_pretrained(FineTuner.BASE_MODEL)
        tokenizer.save_pretrained(FineTuner.MODEL_PATH)

    @staticmethod
    def tokenize(tokenizer, prompt, add_eos_token=True, cutoff_len=512):
        """
        Tokenizes input using provided tokenizer.
        """
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    @staticmethod
    def generate_and_tokenize_prompt(tokenizer, data_point):
        """
        Generates and tokenizes a prompt.
        """
        return FineTuner.tokenize(tokenizer, data_point["text"])

    @staticmethod
    def prepare_data(tokenizer, data, val_set_size):
        """
        Prepares data for training and evaluation.
        """

        def _prepare(dataset):
            return dataset.map(
                lambda x: FineTuner.generate_and_tokenize_prompt(tokenizer, x)
            )

        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            return _prepare(train_val["train"]), _prepare(train_val["test"])
        else:
            return _prepare(data["train"]), None

    def train_model(
        self,
        data,
        output_dir="./lora-alpaca",
        eval_steps=20,
        save_steps=20,
        batch_size=16,
        micro_batch_size=2,
        max_steps=200,
        learning_rate=3e-4,
        val_set_size=100,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        group_by_length=True,
        resume_from_checkpoint=None,
    ):
        """
        Fine-tunes the model with given parameters.
        """
        gradient_accumulation_steps = batch_size // micro_batch_size
        model = LlamaForCausalLM.from_pretrained(self.base_model)
        tokenizer = LlamaTokenizer.from_pretrained(self.base_model, add_eos_token=True)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config).to(self.device)
        if resume_from_checkpoint:
            checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )
                resume_from_checkpoint = False
            if os.path.exists(checkpoint_name):
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
        train_data, val_data = self.prepare_data(tokenizer, data, val_set_size)
        trainer = Trainer(
            model=model.to(self.device),
            train_dataset=train_data,
            eval_dataset=val_data,
            args=TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                max_steps=max_steps,
                learning_rate=learning_rate,
                fp16=False,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_steps if val_set_size > 0 else None,
                save_steps=save_steps,
                output_dir=output_dir,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=None,
                group_by_length=group_by_length,
                report_to="none",
                run_name=None,
                use_ipex=True,
            ),
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.config.use_cache = False
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        model.save_pretrained(output_dir)

    def finetune(
        self,
        input_data_path: str,
        batch_size: Optional[int] = 16,
        micro_batch_size: Optional[int] = 2,
        max_steps: Optional[int] = 200,
        learning_rate: Optional[float] = 3e-4,
        val_set_size_ratio: Optional[float] = 0.1,
    ):
        data = load_dataset("json", data_files=input_data_path)
        num_samples = len(data["train"])
        val_set_size = ceil(val_set_size_ratio * num_samples)
        self.train_model(
            data=data,
            val_set_size=val_set_size,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
        )


def main(input_data,batch_size=16,micro_batch_size=2,max_steps=200,learning_rate=3e-4,val_set_size_ratio=0.1):
    """
    Main function to initiate the finetuning process.
    """
    finetuner = FineTuner()
    finetuner.download_models()
    finetuner.finetune(input_data, batch_size, micro_batch_size, max_steps, learning_rate, val_set_size_ratio)


if __name__ == "__main__":
    Fire(main)