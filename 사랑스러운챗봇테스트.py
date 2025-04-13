from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 기본 모델 로드
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# LoRA 어댑터 로드의
adapter_name = "HongKi08/14B_KOR_MED"
model = PeftModel.from_pretrained(base_model, adapter_name)

model.eval()


input_text = "당신은 의료 전문가입니다 의료적 답변을 해 주세요"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

