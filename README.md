# LLM_FineTurning

# HAI 내에서 진행하는 해당 프로젝트에 대한 문서임.
1. 서버 메뉴얼을 필독 후 원리를 이해 한 이후에 접근 하는 것을 추천하는 바임
2. 기본적으로 사용하는 Unsloth과 그에 딸려오는 Transformer 등 프로그램이 사용하는 CUDA의 원리를 이해하고 시작하여야 함
3. 사용하게 되는 Model 은 DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit 이며 해당 모델은 R1의 증류 모델이고 32B이며 4Bit 양자화가 된 모델이라는 뜻임

# DeepSeek R1 정리
DeepSeek의 추론엔진  

https://arxiv.org/pdf/2502.07316
흔히 아는 LLM의 추론 방식은 총 두가지임
수학문제 데이터 : 문제를 주고 이를 수학적으로 추론해가며 과정을 학습
Chain-of-Thought 데이터 : 사람 생각처럼 해결하는데 이를 문장 설명함 을 학습함

DeepSeek 는 이걸 Codel/O 로 학습을 시킨게 차이점임 (추론엔진)
맥락에서의 논리 흐름 계획 상태공간 탐색 의사결정 트리 순회 등을 특징으로 봄
즉 이걸 통해서 기본 LLM들의 코드 문법으로 부터의 분리를 해내게 되었음.

위에서 기술한 전통적 학습법은 결국 일반화의 한계가 존재함

Input/Output 을 예측하여서 CoT 추론을 사용함
이는 기존 문법에서 벗어나서 구조화된 추론을 가능하게 함 

그러면서 동시에 절차적 엄밀성을 유지하여 다양한 패턴의 학습이 가능함
허나 이는 수학적 추론, 코드 구성 등에서 이점을 가지나 그 외의 추론에서는 성능 저하
그 제외되는 추론의 항목은 일반적인 자연어 CoT 데이터는 추론이 이미 명확하게 정의되어 있지 않아 일관성이 부족할 수 있음
그리고 수학 문제는 특정한 유형의 추론에만 집중하기 때문에 더 다양한 논리적 사고를 학습하기 어렵다. 라는 문제를 가진다

기존 방법과의 차별점
기존 CoT는 사람이 작성한 데이터를 사용하여 일관성이 없을 수 있고 편향된 성능
하지만 CodeI/O에서는 i/o 관계를 이용하여 CoT 자동 생성
코드 실행과정은 논리적으로 엄밀 다양한 추론 패턴을 학습이 가능함

문제 제기
기존의 LLM은 수학 문제 해결이나 코드 생성과 같은 특정 분야에서는 풍부한 학습 데이터를 통해 성능을 향상시켜 옴 허나 논리적 추론, 과학적 추론, 상징적 추론 등 다양한 추론 학습에서는 학습 데이터의 부족과 단편성으로 성능 향상에 한계가 있었습니다.

제안된 방법
Code i/o 를 이용하여 코드에 내제된 다양한 추론 패턴을 체계적으로 추출하여 자연어 형태의 i/o 예측 데이터로 변환하는 방법을 제안하였음
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(factorial(3))

이걸 자연어로 이해시킨다고 생각하면

factorial(3)을 호출한다.
3 != 0이므로, 3 * factorial(2)을 계산해야 한다.
factorial(2)을 호출한다.
2 != 0이므로, 2 * factorial(1)을 계산해야 한다.
factorial(1)을 호출한다.
1 != 0이므로, 1 * factorial(0)을 계산해야 한다.
factorial(0)을 호출한다.
0 == 0이므로, factorial(0) = 1을 반환한다.
이를 이용해, factorial(1) = 1 * 1 = 1이 된다.
이를 이용해, factorial(2) = 2 * 1 = 2가 된다.
이를 이용해, factorial(3) = 3 * 2 = 6이 된다.
이를 통하여 모델이 코드의 구문에 얽매이지 않은 채 논리적 흐름 계획, 상태 공간 탐색,의사 결정 트리 순회, 모듈식 분해 등의 범용 추론 능력을 학습이 가능하게 함.

그래프로 보자
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

이걸 CodeI/O 에서 접근하는 방식으로 설명합니다
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

(Root) x < 5 ?
        /        \
    Yes          No
    /             \
 x < 3 ?         x < 8 ?
  /    \         /      \
 A      B       C        D

학습 과정
모델은 주어진 code and TestCase 를 기반으로 자연어 형태의 CoT 추론을 통하여 
i/o 를 예측하도록 훈련이 됩니다. 이 과정에서 모델은 구체적인 구문에서 벗어나, 절차적 엄밀성을 유지하면서도 다양한 추론 패턴을 내재화 하게 됩니다.

검증 및 성능 향상 
에측된 i/o 을 실제 정답과 비교하거나 code를 재실행해서 결과를 검증함
해당 과정을 통하여 CoT를 다단계로 수정하는 CodeI/O++ 방법도 추가적으로 제시함
