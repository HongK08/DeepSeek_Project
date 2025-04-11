# LLM_FineTurning

# HAI 내에서 진행하는 해당 프로젝트에 대한 문서임.
1. 서버 메뉴얼을 필독 후 토치 또는 텐서의 원리를 이해 한 이후에 접근 하는 것을 추천하는 바임
2. 기본적으로 사용하는 Unsloth과 그에 딸려오는 Transformer 등 프로그램이 사용하는 CUDA의 원리를 이해하고 시작하여야 함
3. 사용하게 되는 Model 은 DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit 이며 해당 모델은 R1의 증류 모델이고 32B이며 4Bit 양자화가 된 모델이라는 뜻임

4. 202032003 길경민


# 참조한 논문
https://arxiv.org/pdf/2502.07316
https://arxiv.org/html/2412.19437v1#S3
https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms 

# DeepSeek R1 정리
DeepSeek의 추론엔진  

흔히 아는 LLM의 추론 방식은 총 두가지 입니다. <br/>
수학문제 데이터 : 문제를 주고 이를 수학적으로 추론해가며 과정을 학습하는 방법 <br/>
Chain-of-Thought 데이터 : 사람 생각처럼 해결 방식 트리를 사용하는 방식

DeepSeek 는 이걸 Codel/O 로 학습을 시킨게 차이점임 (추론엔진)
맥락에서의 논리 흐름 계획 상태공간 탐색 의사결정 트리 순회 등을 특징으로 봄
즉 이걸 통해서 기본 LLM들의 코드 문법으로 부터의 분리를 해내게 되었음.

결국은 위에서 기술한 전통적 학습법은 결국 일반화의 한계가 존재하게 됩니다.

Input/Output 을 예측하여서 CoT 추론을 사용하게 된다면,
이는 기존 문법에서 벗어나서 구조화된 추론을 가능하게 됩니다.

또한 동시에 절차적 엄밀성을 유지하여 다양한 패턴의 학습이 가능하게 됩니다.
허나 이는 수학적 추론, 코드 구성 등에서 이점을 가지나 그 외의 추론에서는 성능 저하를 가져오는 문제가 있습니다.
그 제외되는 추론의 항목은 일반적인 자연어 CoT 데이터는 추론이 이미 명확하게 정의되어 있지 않아 일관성이 부족할 수 있으며,
그리고 수학 문제는 특정한 유형의 추론에만 집중하기 때문에 더 다양한 논리적 사고를 학습하기 어렵다. 라는 문제를 가지는 것 입니다.

기존 방법과의 차별점
기존 CoT는 사람이 작성한 데이터를 사용하여 일관성이 없을 수 있고 편향된 성능을 가지는 문제가 있었으나,
하지만 CodeI/O에서는 i/o 관계를 이용하여 CoT 자동 생성을 하게 되었고,
코드 실행과정은 논리적으로 엄밀 다양한 추론 패턴을 학습이 가능해졌습니다.

문제 제기
기존의 LLM은 수학 문제 해결이나 코드 생성과 같은 특정 분야에서는 풍부한 학습 데이터를 통해 성능을 향상시켜 옴 허나 논리적 추론, 과학적 추론, 상징적 추론 등 다양한 추론 학습에서는 학습 데이터의 부족과 단편성으로 성능 향상에 한계가 있었습니다.

제안된 방법
Code i/o 를 이용하여 코드에 내제된 다양한 추론 패턴을 체계적으로 추출하여 자연어 형태의 i/o 예측 데이터로 변환하는 방법을 제안하였는데

    def factorial(n):
            if n == 0:
                return 1
            return n * factorial(n-1)
            
    print(factorial(3))

이걸 자연어로 이해시킨다고 생각해야 합니다 이는 하단의 글처럼 풀이가 가능합니다.

factorial(3)을 호출한다. <br/>
3 != 0이므로, 3 * factorial(2)을 계산해야 한다. <br/>
factorial(2)을 호출한다. <br/>
2 != 0이므로, 2 * factorial(1)을 계산해야 한다. <br/>
factorial(1)을 호출한다. <br/>
1 != 0이므로, 1 * factorial(0)을 계산해야 한다. <br/>
factorial(0)을 호출한다. <br/>
0 == 0이므로, factorial(0) = 1을 반환한다. <br/>
이를 이용해, factorial(1) = 1 * 1 = 1이 된다. <br/>
이를 이용해, factorial(2) = 2 * 1 = 2가 된다. <br/>
이를 이용해, factorial(3) = 3 * 2 = 6이 된다. <br/>

이를 통하여 모델이 코드의 구문에 얽매이지 않은 채 논리적 흐름 계획, 상태 공간 탐색,의사 결정 트리 순회, 모듈식 분해 등의 범용 추론 능력을 학습이 가능해지는 이야기 입니다.

그림으로 설명합니다.

            ROOT
             |
           [x < 5]
           /    \
       YES      NO
       /         \
    [x < 3]      [x < 8]
     /    \      /    \
    A      B    C      D

DFS/BFS 같은 순회
Root → Left → Left (A) → Backtrack → Right (B) → Backtrack → Right (C) → Right (D)
Root → Left → Right → Left Child (A, B) → Right Child (C, D)

이걸 CodeI/O 에서 접근하는 방식으로 설명합니다.

    def decision_tree(x):
        if x < 5:
            if x < 3:
                return "A"
            else:
                return "B"
        else:
            if x < 8:
                return "C"
            else:
                return "D"
    
    print(decision_tree(2))  # "A"
    print(decision_tree(4))  # "B"
    print(decision_tree(6))  # "C"
    print(decision_tree(9))  # "D"

트리 구조로 보게 된다면.

             x < 5 ?
            /        \
        Yes          No
        /             \
     x < 3 ?         x < 8 ?
     /    \         /      \
    A      B       C        D

학습 과정 <br/>
모델은 주어진 code and TestCase 를 기반으로 자연어 형태의 CoT 추론을 통하여 
i/o 를 예측하도록 훈련이 됩니다. 이 과정에서 모델은 구체적인 구문에서 벗어나, 절차적 엄밀성을 유지하면서도 다양한 추론 패턴을 내재화 하게 됩니다.

예시 상황 <br/>
문제 : 주어진 정수 N에 대해 1부터 까지의 모든 수 중에서 홀수만 출력하는 함수를 작성하십시오. 그리고 리스트로 출력하십시오

1안 <br/>
우선 input을 5로 줍니다
그러면 당연하게도 [1,3,5] 라는 값을 반환하게 될 겁니다.

2안<br/>
input = 5 로 동일합니다.

문제 이해 = 상단과 동일합니다
입력 분석 : 입력값은 5입니다. -> 이는 함수가 1부터 input값 까지 수를 검사해야 한다는 뜻 
출력 예측: 1,3,5 는 홀수니까 (//2) 해서 나누면 당연히 홀 짝 구분이 될 테니까 이를 List에 append 해주는 과정을 합니다

    def get_odds(N):
    return [x for x in range(1, N+1) if x % 2 == 1]

즉 이 함수는 1부터 N(input) 까지의 수를 반복하고 홀수만 얻어내고 List로 Return 해 주는 함수입니다.

이런 방식으로 두가지로 서로 나누어 학습하는 과정을 가지는 원리라고 정리가 가능합니다.

검증 및 성능 향상 <br/>
에측된 i/o 을 실제 정답과 비교하거나 code를 재실행해서 결과를 검증함
해당 과정을 통하여 CoT를 다단계로 수정하는 CodeI/O++ 방법도 추가적으로 제시합니다.

# 정리
제안 된 CodeI/o 방식은 상징적, 과학적, 논리적, 수학적 추론 작업에서 일관된 성능의 향상을 확인하였습니다.
이는 다양한 추론 패턴을 자연어 형태로 학습하여 모델의 전반적인 추론 능력이 향상된다 라는 내용입니다

# FINE_TURNING IN HAI
LLM을 학습시키는 방법은 지도학습/보강학습/보상학습 등의 방법론이 여러가지가 있습니다.<br/>
하지만 이번 파인튜닝에서 사용할 부분은 지도학습 입니다. 즉 기본 DataSet에 가져온 Data를 Push 하여 Prompt를 주는 상황에서의 답을 더 정확하게 만드는 행위 입니다.<br/>
이를 위해서 우리는 HuggingFace 라는 Ai 플랫폼을 활용 할 것이며 이곳에서 디스틸 된 모델을 다운받고 거기에 원하는 1차 데이터셋을 튜닝 + 딥시크의 부족한 언어 성능을 한국어 셋으로 튜닝<br/>
그 이후에 한국 의료정보와 기타 필요한 DataSet을 Turning 후에 이를 Quant 하여 최종 모델 축소까지를 목표로 하는 프로젝트입니다<br/>

1. 사용을 위한 한국어 데이터셋의 불러오기 과정 코드
   
        import os
        import pandas as pd
        
        dir_path = './kor_eng' # 엑셀 파일이 저장된 디렉터리
        files = os.listdir(dir_path)
        print(files)
        
        merge_df = pd.DataFrame()
        
        for file in files:
            df = pd.read_excel(f'{dir_path}/{file}')
            df = df[['원문', '번역문']]
            merge_df = pd.concat([merge_df, df])
        
        merge_df.columns = ['ko', 'en']
        merge_df.to_csv('./dataset.csv', index=False)

해당 코드를 사용하기 전에 우선 AI허브에서 원하는 번역 파일을 다운 받아야만 합니다. <br/><br/>
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=126 <br/>

이후에 해당 데이터의 헤더값을 찿아서 코드에 넣고 지정해주면서 학습을 시키면 됩니다 이는 모든 데이터셋 학습 방법과 상기 동일합니다.  
하지만 HuggigFace(HF) 에서 import 해서 사용하는 법도 있습니다 추후에 전체 소스 코드를 보며 설명을 이어 가도록 하겠습니다.  

이제 제가 작성한 코드를 보며 기본적인 4bit+디스틸 모델을 사용해서 사용하는 법을 설명하겠습니다.  

1. 기본적인 환경 설정  
    Unsloth 이 부분은 그냥 bash에서 pip install unsloth 하시면 종속성 라이브러리까지 모두 설치됩니다.  
    이후 transformers, wandb 등을 설치하며 llama.cpp는 github 공식 라이브러리 참고하시어 설치하십시오.  
    Huggingface도 마찬가지 입니다. 물론 가장 기초적인 Torch는 서버 설정 메뉴얼 참고하셔야 합니다.  
   
2. 이후 설정이 되셨다면  
    HF(HuggingFace) 로그인을 하셔야 합니다. 이때 HF에 가입하시고 개인 토큰 키 발급받으셔야 진행 가능합니다.
    Token 으로 login 호출하셔서 다음 seq 길이의 지정 dtype 등 지정과 load_in_4bit 은 사용하시는 모델에 따라 지정하십시오.
    튜닝 전 prompt_style 지정해서 튜닝 전 결과값 보셔도 좋습니다. 이때 지정된 인스트럭션 질문 답변 등 형식을 유지하십시오.
    그리고 질문을 주면 얘가 답변을 해 줄겁니다 기본적인 LLM 내에서 정제 된 R1이 주는 답변입니다.
    이후 Train 후에 나올 Prompt의 양식도 상단과 같이 지정해야 합니다 이는 사용 할 모델의 구조와 학습 데이터셋을 고려하십시오.

3. 데이터 셋의 처리 준비  
    이제 상단에서 지정한 프롬프트 스타일을 formatting_prompts_func() 라는 항목에서 전부 지정해 주어야만 합니다.  
    이곳에서 기본적인 구조를 정의하시고 난 다음에야 저희는 데이터셋을 불러오고 그 내부의 헤더값을 사용하여 학습을 시킬겁니다. (peft 사용함)  
    그러면 이제 원하는 데이터셋의 구조를 모두 파악하였다는 가정 하에 학습을 진행 해 보도록 하겠습니다.  
    from dataset import load_dataset을 하여 HF상에 있는 DataSet을 사용한다고 생각하시면 됩니다  
        1. 가져 올 데이터셋의 직접 주소 또는 HF 라이브러리 명 + 데이터 셋 이름  
        2. dataset.ap을 지정해주는대 이때 그 값은 위에서 정의한 formatting_prompt_func 을 쓰게 되는겁니다.  
        3. dataset에서 꺼내 쓸 것 즉 첫 읽는 위치를 지정하면 끝납니다  
    이제 wandb를 비활성화(학습에 지장을 줌) 을 환경변수에 ~/.bashrc에 지정하시던지 코드내에서 하드지정 꼭 하십시오 중요합니다.  
    그리고 클래스 아래에 생성자 만드셔서 전부 하나하나 지정해 주어야 합니다 이는 코드 참고하시면 좋습니다.  
    그리고 토크나이징 후에 return으로 반환까지 해주면 정리가 됩니다.  

4. Train    
    설정이 모두 완료되었으니 이제 학습을 시킬 차례입니다.
    여기서 bach_size/gradient 등 값을 모두 원하는대로 지정하셔야 하는데 이때 이 값은 현재 Server의 GPU V-RAM 용량을 고려하시며 지정하십시오.  
    간단하게 이 값이 높아질수록 GPU의 사용율이 높아짐과 동시에 Train에서 Epoch에도 영향을 주게 됩니다.  
    그리고 learnig_rate= 값도 본인이 고려햐여 지정, 연산 방식은 bf16/fp16이 있는데 구형 GPU는 fp16만 지원하나 Server의 4090은 bf16을 지원합니다.  
    그리고 그 외의 값은 구글링을 통하여 왜 이렇게 되고 어떤 방식이 내 모델에 적합한가를 생각하시고 Seed값은 랜덤으로 두시던지 자유입니다.  

    이후 trainer 호출해서 돌려주면 됩니다.  
    그럼 당연히 나온 모델이 바뀐 값을 잘 추론해주는가? 이걸 확인하고 싶으실겁니다.  
    마찬가지로 상단의 지정하신 prompt 를 그대로 사용하시며 인스트럭션은 Dataset 제공자의 권장사항을 따르는 것이 좋습니다.  
    그리고 이때 아웃풋이나 인덱스값은 당연히 받고싶은 prompt 에 맞게 지정하셔야 하며 to("cuda") 지정하시어 사용하십시오  

    
    그 이후 로컬에 quant 하시어 저장하시거나 HF에 create_repo/repo_name 지정하시어 push 하시면 기본적인 파인튜닝이 끝나게 됩니다.
    
        new_model_local = "<name>"
        model.save_pretrained(new_model_local)
        tokenizer.save_pretrained(new_model_local)
        
        model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",) 
        model.push.to.hub_gguf(repo.name,tokenizer, quantization_method="<llama.cpp 문서 참조하여 기입>")


# 주석이 모두 달린 첫 베이스 코드
    
    from huggingface_hub import login
    from getpass import getpass
    
    #토큰 입력 (비밀번호처럼 입력됨)
    hf_token = '<YOUR_TOKEN>'
    login(token=hf_token) #귀찮아서 그냥 강제로 지정해버렸습니다.

    #모델 로드
    from unsloth import FastLanguageModel
    
    #모델 설정값 정의
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    # 모델과 토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",  # 정확한 모델명
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=None
    )
    
    print("살려주세요!")
    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
    
    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
    Please answer the following medical question.
    
    ### Question:
    {}
    
    ### Response:
    <think>{}"""

    question = "A 55-year-old extremely obese man experiences weakness, sweating, tachycardia, confusion, and headache when fasting for a few hours, which are relieved by 
     eating. What disorder is most likely causing these symptoms?"


    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Response:")[1])

    #DataSet Loading
    
    train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
    
    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
    Please answer the following medical question.
    
    ### Question:
    {}
    
    ### Response:
    <think>
    {}
    </think>
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


    def formatting_prompts_func(examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    #Model setup

    from datasets import load_dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[0:700]",trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    dataset["text"][0]

    model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    )

    # Train

    import os
    import wandb
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    
    #wandb 비활성화
    os.environ["WANDB_DISABLED"] = "true"
    wandb.init(mode="disabled")

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
    
            # 토크나이징
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
    
            # labels 추가 (input_ids와 동일하게 설정)
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()  # labels 추가
            }
    
    processed_dataset = CustomDataset(dataset, tokenizer, max_seq_length)
    
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
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
        )
        
    trainer.train()
    
        # After Fine_Turning
    
        question = "A 55-year-old extremely obese man experiences weakness, sweating, tachycardia, confusion, and headache when fasting for a few hours, which are relieved by eating. What disorder is most likely causing these symptoms?"
    
    
        FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
        inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
        )
        response = tokenizer.batch_decode(outputs)
        print(response[0].split("### Response:")[1])
    
        new_model_local = "<name>"
        model.save_pretrained(new_model_local)
        tokenizer.save_pretrained(new_model_local)
        
        model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",)
    
        from huggingface_hub import login, create_repo
        import os
        
        # 1. 환경변수 설정
        os.environ["HUGGINGFACE_TOKEN"] = "hf_your_token"
        login(token=hf_token)  # 토큰으로 로그인
        
        # 2. 새 저장소 생성 후 업로드
        repo_name = "<YOYR_REPO/NAME>"
        create_repo(repo_name)
        model.push_to_hub_gguf(repo_name, tokenizer, quantization_method="q4_k_m")

# 이렇게 하여 1차 튜닝을 완료합니다.
    
Q4로 저장했으나 저희는 이후 값 비교를 위해 한국어 데이터 셋을 추가적으로 학습시킬 예정입니다


    from numba import cuda
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    from huggingface_hub import login
    from unsloth import FastLanguageModel
    import os
    import wandb
    
    # CUDA 리셋 (메모리 초기화)
    device = cuda.get_current_device()
    device.reset()
    
    # Hugging Face 로그인
    hf_token = "your_huggingface_token"
    login(token=hf_token)
    
    # 모델 설정
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    # 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=None
    )
    print("모델 로딩 완료!")
    
    # 데이터셋 로딩
    dataset = load_dataset("MarkrAI/KoCommercial-Dataset", split="train")
    print("데이터셋 로딩 완료!")
    
    # 데이터셋 변환 함수
    def format_data(example):
        instruction = example.get("instruction", "다음 광고 문구를 분석하세요.")
        input_text = example.get("input", "")
        output_text = example.get("output", example.get("text", ""))  # 'text' 필드도 고려
    
        return {
            "text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        }
    
    # 데이터 변환 적용
    formatted_dataset = dataset.map(format_data)
    
    # 데이터셋 클래스 정의
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
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
    
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            }
    
    # 변환된 데이터셋을 CustomDataset으로 처리
    processed_dataset = CustomDataset(formatted_dataset, tokenizer, max_seq_length)
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=1000,
        learning_rate=2e-5,
        fp16=False,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # wandb 비활성화
    os.environ["WANDB_DISABLED"] = "true"
    wandb.init(mode="disabled")
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # 훈련 시작
    trainer.train()
    
    # 훈련 완료 후 모델 저장
    model.push_to_hub("HongKi08/Korean-Qwen14B", tokenizer, quantization_method="q4_k_m")
    print("모델 업로드 완료!")

이 떄 양자화 조건이라던가 그런 구문은 상황에 맞게 변경하여 로컬에 저장하던 업로드하던 개인의 선택입니다.

    repo_name = "HongKi08/SanHak"
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)






#
    import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login
import subprocess
import shutil

# Hugging Face 로그인
hf_token = "hf_..."  # 너의 토큰으로 바꿔줘
login(hf_token)

# 설정
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
lora_model_id = "HongKi08/14B_KOR_MED_VAL"
merged_dir = "./merged_14b_kor_med"
gguf_dir = "./gguf_model"
repo_id_transformers = "HongKi08/14B_KOR_MED_MERGED"
repo_id_gguf = "HongKi08/14B_KOR_MED_GGUF"

# 1. 병합
print("▶️ 병합 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype="auto"
)
model = PeftModel.from_pretrained(base_model, lora_model_id)
merged_model = model.merge_and_unload()

# 2. 저장
print("💾 저장 중...")
tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

# 3. Hugging Face에 업로드 (transformers format)
print("⬆️ Hugging Face 업로드 중 (transformers)...")
api = HfApi()
api.create_repo(repo_id_transformers, exist_ok=True)
merged_model.push_to_hub(repo_id_transformers)
tokenizer.push_to_hub(repo_id_transformers)

# 4. GGUF 변환 (llama.cpp 사용)
print("🔁 GGUF 변환 중...")
gguf_script = "./llama.cpp/convert.py"
if not os.path.exists("llama.cpp"):
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])

subprocess.run([
    "python3", gguf_script,
    merged_dir,
    "--outfile", f"{gguf_dir}/model.gguf"
])

# tokenizer도 복사
shutil.copytree(os.path.join(merged_dir, "tokenizer_config.json"), os.path.join(gguf_dir, "tokenizer_config.json"), dirs_exist_ok=True)

# 5. Hugging Face에 업로드 (gguf)
print("⬆️ Hugging Face 업로드 중 (GGUF)...")
api.create_repo(repo_id_gguf, exist_ok=True)
api.upload_folder(
    folder_path=gguf_dir,
    repo_id=repo_id_gguf,
    repo_type="model"
)

print("✅ 모든 과정 완료!")


학습 -> 다시 불러옴 -> 학습 의 과정을 진행하고 싶다면 이런식으로 Transformer 가 사용 가능하도록 지정하여 push 해줍니다.

이 과정을 반복하면서 각 학습 진행도에 따른 벤치마크의 결과를 보도록 하겠습니다.

