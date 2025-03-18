import os  # 환경 변수 설정을 위한 라이브러리
import wandb  # 학습 모니터링을 위한 라이브러리 (여기서는 비활성화됨)
from huggingface_hub import login  # Hugging Face 허브 로그인
from getpass import getpass  # 안전한 패스워드 입력을 위한 라이브러리

# Hugging Face API 토큰 설정 및 로그인
hf_token = '<>'
login(token=hf_token)

# 모델 설정 변수
dtype = None  # 데이터 타입 (기본값 사용)
max_seq_length = 2048  # 모델이 처리할 최대 시퀀스 길이
load_in_4bit = True  # 4비트 양자화 여부

# Unsloth 라이브러리를 활용하여 4bit 양자화된 모델 로드
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import create_repo, HfApi
import torch

# 모델 및 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",  # 모델 이름
    max_seq_length=max_seq_length,  # 최대 시퀀스 길이
    dtype=dtype,  # 데이터 타입 설정 (기본값 사용)
    load_in_4bit=load_in_4bit,  # 4비트 양자화 여부
    token=None  # 추가 토큰 설정 없음
)

# 프롬프트 스타일 설정 (질문과 답변 형식 지정)
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a language expert who is good at Korean,
also a medical professional with advanced knowledge of clinical reasoning, diagnosis, and treatment planning.
Please answer the following medical questions.

### Question:
{}

### Response:
<think>{}"""

# 테스트용 질문 설정
question = "A 55-year-old extremely obese man experiences weakness, sweating, tachycardia, confusion, and headache when fasting for a few hours, which are relieved by eating. What disorder is most likely causing these symptoms?"

# 모델을 추론(inference) 모드로 전환
FastLanguageModel.for_inference(model)

# 입력 데이터 토큰화 및 GPU로 이동
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# 모델을 이용하여 답변 생성
outputs = model.generate(
    input_ids=inputs.input_ids,  # 입력 데이터
    attention_mask=inputs.attention_mask,  # 마스크 설정
    max_new_tokens=1200,  # 최대 생성 토큰 수
    use_cache=True,  # 캐시 사용 (속도 향상)
)

# 모델 출력 디코딩 후 결과 출력
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

# 학습용 데이터셋 로드 (700개 샘플 사용)
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:700]", trust_remote_code=True)

# EOS 토큰 추가
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """데이터셋을 프롬프트 형식에 맞게 변환하는 함수"""
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# 데이터셋 변환
dataset = dataset.map(formatting_prompts_func, batched=True)

# LoRA 기반 파인튜닝 설정
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA 랭크 설정
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,  # LoRA 알파 값 설정
    lora_dropout=0,  # LoRA 드롭아웃 없음
    bias="none",  # 편향 사용 안 함
    use_gradient_checkpointing="unsloth",  # 긴 컨텍스트 최적화를 위해 Unsloth 방식 사용
    random_state=3407,  # 랜덤 시드 고정
    use_rslora=False,  # rslora 사용 안 함
    loftq_config=None,  # 추가 설정 없음
)

# wandb 비활성화
os.environ["WANDB_DISABLED"] = "true"
wandb.init(mode="disabled")

# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        encodings = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

# 데이터셋 처리
processed_dataset = CustomDataset(dataset, tokenizer, max_seq_length)

# 학습 설정
training_args = TrainingArguments(
    output_dir="outputs",  # 출력 디렉터리
    per_device_train_batch_size=4,  # 배치 크기
    gradient_accumulation_steps=12,  # 그래디언트 누적 스텝
    warmup_steps=5,  # 웜업 스텝 수
    max_steps=80,  # 최대 학습 스텝 수
    learning_rate=2e-4,  # 학습률
    fp16=False,  # 16-bit 연산 비활성화
    logging_steps=10,  # 로그 출력 간격
    optim="adamw_torch",  # 옵티마이저 설정
    weight_decay=0.01,  # 가중치 감쇠
    lr_scheduler_type="linear",  # 학습률 스케줄러
    seed=3407,  # 시드 고정
    report_to="none",  # 로깅 툴 비활성화
    remove_unused_columns=False,  # 불필요한 컬럼 제거 안 함
)

# 트레이너 설정 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)
trainer.train()

# Hugging Face에 모델 업로드
repo_name = "</>"
model.push_to_hub_gguf(repo_name, tokenizer, quantization_method="q6_k")
