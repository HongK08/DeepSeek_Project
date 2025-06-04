# LLM Fine-Tuning in HAI Project

## 프로젝트 개요

HAI 내부에서 진행하는 LLM 파인튜닝 프로젝트입니다. 주로 한국어 의료 데이터를 활용해 다단계 파인튜닝을 수행하며, `Unsloth` 기반의 QLoRA+LoRA 학습 구조를 사용합니다.

* GPU: 4090 / H100 기준 (bf16 및 4bit 양자화 지원)
* 모델: `DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit`

---

## 참고 논문 및 자료

* [https://arxiv.org/pdf/2502.07316](https://arxiv.org/pdf/2502.07316)
* [https://arxiv.org/html/2412.19437v1#S3](https://arxiv.org/html/2412.19437v1#S3)
* [https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)

---

## DeepSeek R1 개요

### 학습 구조

DeepSeek은 기존 CoT(Chain-of-Thought) 방식의 한계를 극복하기 위해, 코드 기반 입력-출력 구조인 `Code I/O` 방식으로 훈련되었습니다.

### 예시: 재귀 함수의 자연어 추론

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(factorial(3))
```

이를 자연어로 풀이:

* 3은 0이 아니므로 3 \* factorial(2)
* 2는 0이 아니므로 2 \* factorial(1)
* 1은 0이 아니므로 1 \* factorial(0)
* factorial(0)은 1
* 결과적으로 3 \* 2 \* 1 = 6

### 구조적 추론 트리 예시

```
         x < 5 ?
        /       \
     Yes         No
    /             \
 x < 3 ?        x < 8 ?
 /     \        /     \
A       B      C       D
```

---

## 파인튜닝 준비

### 환경 설정

* HuggingFace 로그인 (Token 필요)
* `unsloth`, `transformers`, `wandb`, `datasets` 설치

### AI Hub 데이터 전처리 예시

```python
import os
import pandas as pd

files = os.listdir('./kor_eng')
merge_df = pd.concat([
    pd.read_excel(f'./kor_eng/{f}')[['원문', '번역문']] for f in files
])
merge_df.columns = ['ko', 'en']
merge_df.to_csv('./dataset.csv', index=False)
```

---

## 프롬프트 템플릿

```text
### Instruction:
You are a medical expert...

### Question:
{}

### Response:
<think>
{}
</think>
{}
```

`formatting_prompts_func()`을 통해 학습 전 데이터에 적용

---

## LoRA 구성

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

---

## 학습 설정

```python
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    remove_unused_columns=False
)
```

---

## 모델 저장 및 업로드

HuggingFace로 업로드:

```python
model.push_to_hub("HongKi08/Korean-Qwen14B", tokenizer, quantization_method="q4_k_m")
```

로컬 저장:

```python
new_model_local = "your_path"
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)
```

16bit 병합 저장:

```python
model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit")
```

---

## 전체 플로우 요약

1. 1차 파인튜닝: `FreedomIntelligence/medical-o1-reasoning-SFT`
2. 2차 파인튜닝: `MarkrAI/KoCommercial-Dataset` + 번역 데이터
3. 매 차수별 `.safetensors` 저장 및 병합
4. 최종 병합된 모델 HuggingFace에 업로드 또는 GGUF로 저장
