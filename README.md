# 🚀 LLM-Guided DRL Routing (LoRA Fine-Tuning)

This project implements **LoRA fine-tuning of a Large Language Model (LLM)** to dynamically assist a **Deep Reinforcement Learning (DRL)** routing system.

The LLM is trained to:
- Generate **reward weights (μ)**
- Assign **privacy levels (ψ)** for each network flow

---

## 🧠 Project Idea

The system works as:

```
Network State → LLM → (μ, ψ) → DRL Agent → Optimized Routing
```

- **LLM** decides priorities (performance, QoS, privacy)
- **DRL** decides how to route traffic

---

## 📁 Project Structure

```
LLM_Tune_ToRa/
│
├── data/
│   └── train.jsonl          # Generated dataset
│
├── adapters/                # LoRA output (ignored in git)
│
├── generate_dataset.py      # Dataset generator
├── train_lora.py            # LoRA training script
├── test_model.py            # Model testing script
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch transformers datasets peft accelerate trl bitsandbytes
```

### 3. Check GPU (optional but recommended)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 Dataset Generation

Generate synthetic dataset:

```bash
python generate_dataset.py
```

Output:

```
data/train.jsonl
```

---

## 🧪 LoRA Training

Run:

```bash
python train_lora.py
```

This will:
- Load TinyLlama model
- Apply LoRA fine-tuning
- Save adapter in:

```
adapters/tinyllama-lora-sdven/
```

---

## 🔍 Model Testing

Run:

```bash
python test_model.py
```

Expected output:

```json
{
  "mu": [0.2, 0.25, 0.2, 0.15, 0.2],
  "privacy": {
    "1": 0.7,
    "2": 0.6
  }
}
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Base Model | TinyLlama-1.1B |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| Output 1 | Reward weights μ |
| Output 2 | Privacy levels ψ per flow |

---

## ⚖️ Constraints

The model outputs must satisfy:

- Sum of μ = 1
- μ ≥ μ_min
- 0 ≤ ψ ≤ 1

> Invalid outputs are corrected automatically during inference.

---

## 🚀 Key Technologies

| Technology | Purpose |
|------------|---------|
| PyTorch | Deep learning framework |
| HuggingFace Transformers | LLM loading & inference |
| PEFT (LoRA) | Parameter-efficient fine-tuning |
| TRL (SFTTrainer) | Supervised fine-tuning |

---

## 📌 Notes

- `.venv` is not included in repo
- `adapters/` is ignored (large files)
- Dataset can be regenerated anytime

---

## 🔥 Future Work

- Integrate with ns-3 simulation
- Real-time LLM inference
- DRL + LLM joint optimization

---

