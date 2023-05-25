import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="intel_extension_for_pytorch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image", lineno=13)

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import intel_extension_for_pytorch as ipex
from fire import Fire
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

MODEL_PATH = "./model"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
CHECKPOINT_PATH = "./lora-alpaca/adapter_model.bin"


class InferenceModel:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
        model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, config)
        checkpoint = torch.load(CHECKPOINT_PATH)
        set_peft_model_state_dict(self.model, checkpoint)
        self.model.to(DEVICE)
        self.model.eval()

    def generate(self, input, max_length=100, **kwargs):
        prompt = self.tokenizer.encode(input, add_special_tokens=False)
        inputs = torch.tensor([prompt], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                do_sample=True,
                max_length=max_length,
                temperature=0.3,
                top_p=0.85,
                top_k=40,
                num_beams=1,
                repetition_penalty=1.2,
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def benchmark(self, num_runs=10):
        benchmark_input = "Tell me about alpacas."
        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.generate(benchmark_input, max_length=100)
            end_time = time.time()

        avg_time = total_time / num_runs
        print(f"Average time over {num_runs} runs: {avg_time} seconds")


def main(user_prompt=None, infer=True, bench=False):
    model = InferenceModel()
    prompts = [
        "Explain the social significance of a ball in Regency England.",
        "Describe the character traits of Elizabeth Bennet.",
        "How does Mr. Darcy's opinion of Elizabeth Bennet change throughout the novel?",
        "What is the role of marriage in Pride and Prejudice?",
        "Describe the relationship between Mr. Bingley and Jane Bennet.",
        "What is the impact of class and status in Pride and Prejudice?",
        "How do the different marriages in the novel comment on society at the time?",
        "What is the importance of first impressions in Pride and Prejudice?",
        "Tell me about alpacas.",
    ]
    if infer:
        if user_prompt is not None:
            prompts = [user_prompt]
        for prompt in prompts:
            print(f"user given prompt: {prompt}")
            start_time = time.time()
            output = model.generate(prompt, max_length=100)
            end_time = time.time()
            print(f"\nbot response: {output}\n")
            print(f"infer time: {end_time - start_time} seconds\n")
    if bench:
        start_time = time.time()
        model.benchmark()
        end_time = time.time()
        print(f"bench time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    Fire(main)
