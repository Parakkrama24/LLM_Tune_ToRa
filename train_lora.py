import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

# ========================
# LoRA CONFIG
# ========================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

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
1. Reward weights mu: 5 values summing to 1
2. Privacy values psi for each flow

Return only valid JSON.

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

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    dataset_text_field="text",
    max_length=1024
)

# ========================
# TRAINER
# ========================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer
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