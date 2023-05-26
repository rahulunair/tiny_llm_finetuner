import gc
import time
import warnings
import statistics


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
        self.model = ipex.optimize(model=self.model.eval(), dtype=torch.bfloat16)
        self.max_length = 100

    def generate(self, input, **kwargs):
        prompt = self.tokenizer.encode(input, add_special_tokens=False)
        inputs = torch.tensor([prompt], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            with torch.xpu.amp.autocast():
                outputs = self.model.generate(
                    input_ids=inputs,
                    do_sample=True,
                    max_length=self.max_length,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    num_beams=5,
                    repetition_penalty=1.2,
                )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def benchmark(self, num_runs=12, num_warmup=3):
        benchmark_input = "Tell me about alpacas."
        times = []
        for _ in range(num_warmup):
            self.generate(benchmark_input)
        for i in range(num_runs):
            start_time = time.time()
            self.generate(benchmark_input)
            end_time = time.time()
            if not i < 2:
                times.append(end_time - start_time)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev_time = statistics.stdev(times) if len(times) > 1 else 0
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {DEVICE}")
        print(f"Data type: FP16")
        print(f"Max tokens: {self.max_length}")
        print(f"Average time over {num_runs} runs: {avg_time} seconds")
        print(f"Min time over {num_runs} runs: {min_time} seconds")
        print(f"Max time over {num_runs} runs: {max_time} seconds")
        print(f"Standard deviation over {num_runs} runs: {std_dev_time} seconds")


def main(user_prompt=None, infer=False, bench=False):
    torch.xpu.empty_cache()
    gc.collect()
    model = InferenceModel()
    prompts = [
        "The social significance of a ball in Regency England.",
        "The character traits of Elizabeth Bennet are.",
        "Mr. Darcy's opinion of Elizabeth Bennet change throughout the novel",
        "The role of marriage in Pride and Prejudice?",
        "The relationship between Mr. Bingley and Jane Bennet.",
        "The impact of class and status in Pride and Prejudice?",
        "The different marriages in the novel comment on society at the time?",
        "The importance of first impressions in Pride and Prejudice?",
        "Let me tell me about alpacas.",
        "python code to find primes using recursion",
    ]
    if infer:
        if user_prompt is not None:
            prompts = [user_prompt]
        for prompt in prompts:
            print(f"user given prompt: {prompt}")
            start_time = time.time()
            output = model.generate(prompt)
            end_time = time.time()
            print(f"\nbot response: {output}\n")
            print(f"infer time: {end_time - start_time} seconds\n")
    if bench:
        model.benchmark()


if __name__ == "__main__":
    Fire(main)
