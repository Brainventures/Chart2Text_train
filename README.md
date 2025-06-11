# Chart2Text_train
Chart2Text 학습 코드입니다.

본 방법론은 ChartAssist의 2Phase 학습을 Follow하고 있습니다.

전체적인 학습 방법은 다음과 같습니다.

1. Table 데이터셋을 Json 형태로 준비 후, qwen_vision_train.py를 통해 학습해 주십시오. 
2. 새로운 데이터셋을 준비해 주십시오.(Table + Description + QA)
3. #1 과정을 통해 학습한 모델을 base model로 하여 새로운 데이터를 학습하여 주십시오.
