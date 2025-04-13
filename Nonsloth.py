import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import login
import os
import wandb

# HF 로그인
hf_token = ""
login(token=hf_token)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 로드 (32B, 4bit)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)
()
# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 데이터셋 로딩
dataset = load_dataset("MarkrAI/KoCommercial-Dataset", split="train")

def format_example(example):
    instruction = example.get("instruction", "지금부터 모든 답변을 한국어로 번역하세요.")
    input_text = example.get("input", "")
    output_text = example.get("output", example.get("text", ""))

    return {
        "text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    }

formatted_dataset = dataset.map(format_example)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
        }

# 하이퍼파라미터
max_seq_length = 2048
batch_size = 1

processed_dataset = CustomDataset(formatted_dataset, tokenizer, max_seq_length)

# wandb 비활성화
os.environ["WANDB_DISABLED"] = "true"
wandb.init(mode="disabled")

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    max_steps=1000,
    warmup_steps=20,
    logging_steps=10,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="none",
    remove_unused_columns=False,
)

# 트레이너 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

# 학습 시작
trainer.train()

# Hugging Face 업로드
repo_name = "HongKi08/DeepSeek-14B-KoLoRA"
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
