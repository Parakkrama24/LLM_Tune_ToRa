import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "adapters/tinyllama-lora-sdven"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

test_input = {
    "sample_id": 9999,
    "num_flows": 3,
    "avg_congestion": 0.72,
    "min_lifetime": 0.35,
    "avg_rwnd": 0.48,
    "privacy_fraction": 0.67,
    "flows": [
        {
            "flow_id": 1,
            "qos": 3,
            "lifetime_ratio": 0.6,
            "congestion": 0.8,
            "rwnd": 0.4
        },
        {
            "flow_id": 2,
            "qos": 2,
            "lifetime_ratio": 0.9,
            "congestion": 0.65,
            "rwnd": 0.5
        },
        {
            "flow_id": 3,
            "qos": 1,
            "lifetime_ratio": 1.2,
            "congestion": 0.3,
            "rwnd": 0.8
        }
    ]
}

prompt = f"""
You are a network optimization assistant.

Given the network state below, generate:
1. Reward weights mu: 5 values summing to 1
2. Privacy values psi for each flow

Return only valid JSON.

### Input:
{json.dumps(test_input, indent=2)}

### Output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== MODEL OUTPUT =====\n")
print(result.split("### Output:")[-1].strip())