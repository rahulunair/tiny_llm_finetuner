### Finetuning openLLAMA on Intel GPUS

A simple finetuner\[^1]\[^2] for LLMs on Intel XPU devices, with which you could finetune the openLLAMA model to sound like your favorite book.

#### Setup and activate conda env

```bash
conda env create -f env.yml
conda activate pyt_llm_xpu
```

**Warning**: If you PyTorch and intel extension for PyTorch already setup, then install peft without dependencies as peft requires PyTorch 2.0(not supported yet on Intel XPU devices.)

#### Generate data

Fetch a book from guttenberg (default: pride and prejudice) and generate the dataset.

```python
python fetch_data.py
```

#### Finetune

```bash
python finetune.py --input_data ./book_data.json --batch_size=64 --micro_batch_size=16 --num_steps=300
```

#### Inferece

```bash
python infer_llm.py
```

For inference, you can either provide a input prompt, or the model will take a default prompt

##### Benchmark Inference

```bash
python infer_llm.py --bench
```

<a name="f1">1</a>: [adapted from: https://github.com/modal-labs/doppel-bot/blob/main/src/finetune.py](https://github.com/modal-labs/doppel-bot/blob/main/src/finetune.py) <a name="f2">2</a>: [adapted from: https://github.com/tloen/alpaca-lora/blob/65fb8225c09af81feb5edb1abb12560f02930703/finetune.py](https://github.com/tloen/alpaca-lora/blob/65fb8225c09af81feb5edb1abb12560f02930703/finetune.py)
