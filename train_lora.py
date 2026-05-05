import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json

# ========================
# CONFIG
# ========================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "adapters/tinyllama-lora-sdven"

# ========================
# LOAD MODEL + TOKENIZER
# ========================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ========================
# LoRA CONFIG
# ========================

lora_config = LoraConfig(
    r=8,                       # rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ========================
# LOAD DATASET
# ========================

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ========================
# PROMPT FORMATTER
# ========================

def format_prompt(example):
    input_data = example["input"]
    output_data = example["output"]

    prompt = f"""
You are a network optimization assistant.

Given the network state below, generate:
1. Reward weights μ (5 values summing to 1)
2. Privacy values ψ for each flow

### Input:
{json.dumps(input_data, indent=2)}

### Output:
{json.dumps(output_data, indent=2)}
"""

    return {"text": prompt}

dataset = dataset.map(format_prompt)

# ========================
# TRAINING CONFIG
# ========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# ========================
# TRAINER
# ========================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=1024
)

# ========================
# TRAIN
# ========================

trainer.train()

# ========================
# SAVE MODEL
# ========================

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. LoRA adapter saved.")