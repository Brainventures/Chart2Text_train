"""
차트 질문 답변을 위한 Vision Language Model 1차 파인튜닝( Table ) Script

이 모듈은 차트 분석 작업을 위한 Qwen2.5-VL 모델의 파인튜닝을 구현합니다.
스크립트는 전체 파인튜닝(Full Fine-tuning)을 사용하여 차트 이미지와 
관련 질문 및 답변을 처리합니다.

작성자: 정원렬
날짜: 2024
"""

import torch
from torch.optim import AdamW
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor , Qwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import gc
import time
from transformers import TrainingArguments, Trainer , EarlyStoppingCallback
from peft import LoraConfig, get_peft_model

from trl import SFTConfig

# RTX Issue
import os

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# RTX Issue


def format_data(sample):
    """
    비전-언어 모델 학습을 위한 입력 데이터를 포맷팅합니다.
    
    원시 데이터셋 샘플을 시스템 메시지, 사용자 쿼리(이미지와 텍스트 포함), 
    어시스턴트 응답을 포함한 대화 형태로 변환합니다.
    
    Args:
        sample (dict): 원시 데이터 샘플로 다음을 포함:
            - image: PIL 이미지 또는 이미지 경로
            - query (str): 차트에 대한 사용자 질문
            - label (str): 예상 답변/응답
    
    Returns:
        list: 역할(system, user, assistant)과 이미지 및 텍스트 구성 요소를 
              포함한 콘텐츠로 구성된 포맷된 대화
    
    Example:
        >>> sample = {"image": chart_img, "query": "추세가 어떻게 되나요?", "label": "증가하고 있습니다"}
        >>> formatted = format_data(sample)
        >>> len(formatted)  # 3개 반환 (system, user, assistant 메시지)
        3
    """
    system_message = """당신은 시각적 데이터(차트 이미지)를 분석하여 질문에 간결한 답변을 제공하는 비전 언어 모델입니다.
    차트 유형(예: 선 그래프, 막대 그래프, 원형 그래프), 색상, 레이블, 텍스트를 기반으로 사용자의 명령에 대응하세요. 
    """
#항상 정확하고 간결하게 대답하며, 추가 설명은 꼭 필요할 때만 포함하세요.
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]

def clear_memory():
    """
    안전한 GPU 메모리 정리 함수
    
    CUDA 초기화 상태를 확인하고 안전하게 메모리를 정리합니다.
    오류 발생 시에도 프로그램이 중단되지 않도록 예외 처리를 포함합니다.
    """
    import gc
    import torch
    
    # Python 가비지 컬렉션 먼저 실행
    gc.collect()
    
    # CUDA가 사용 가능하고 초기화되었는지 확인
    if torch.cuda.is_available():
        try:
            # GPU 메모리가 할당되어 있는지 확인
            if torch.cuda.memory_allocated() > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
        except RuntimeError as e:
            print(f"CUDA 동기화 중 오류 발생: {e}")
            print("GPU 메모리 정리를 건너뜁니다.")
    else:
        print("CUDA를 사용할 수 없습니다.")
    
    # 마지막으로 한 번 더 가비지 컬렉션
    gc.collect()

def collate_fn(examples):
    """
    비전-언어 모델 학습을 위한 사용자 정의 데이터 콜레이션 함수입니다.
    
    대화 예제 배치를 다음과 같이 처리합니다:
    1. 대화를 포맷하기 위해 채팅 템플릿 적용
    2. 이미지에서 비전 정보 추출 및 처리
    3. 텍스트 토큰화 및 이미지 입력 준비
    4. 학습을 위한 적절한 라벨 생성 (패딩 토큰 마스킹)
    
    Args:
        examples (list): format_data()에서 반환된 포맷된 대화 예제 목록
    
    Returns:
        dict: 다음을 포함한 배치 딕셔너리:
            - input_ids: 토큰화된 입력 시퀀스
            - attention_mask: 패딩을 위한 어텐션 마스크
            - pixel_values: 처리된 이미지 텐서
            - labels: 학습을 위한 타겟 라벨 (패딩 토큰은 -100)
    
    Note:
        processor.tokenizer.pad_token_id를 사용하여 패딩 토큰을 식별하고
        해당 라벨을 -100으로 설정합니다 (손실 계산에서 무시됨).
    """
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch


def main():
    """
    Vision Language Model 파인튜닝을 위한 메인 학습 파이프라인입니다.
    
    이 함수는 완전한 학습 프로세스를 조율합니다:
    1. 데이터 로딩 및 전처리
    2. 모델 및 프로세서 초기화
    3. 학습 설정 및 실행
    4. 모델 저장
    
    스크립트는 차트 질문-답변 데이터에 대해 Qwen2.5-VL 모델을 
    전체 파인튜닝(Full Fine-tuning)으로 학습시킵니다.
    
    설정값:
        - 모델: Qwen2.5-VL-3B-Instruct
        - 학습 에포크: 2
        - 배치 크기: 2 (그래디언트 누적 포함)
        - 학습률: 3e-5
        - 옵티마이저: Adafactor
        - 정밀도: BFloat16
    
    Raises:
        RuntimeError: CUDA를 사용할 수 없거나 GPU 메모리가 부족한 경우
        FileNotFoundError: 데이터셋 경로가 유효하지 않은 경우
    """
    clear_memory()

    nia_path = '/workspace/data_moder/GPT_TABLE.json'
    dataset = load_dataset('json',  data_files = nia_path )

    # 데이터셋 분할
    dataset_num = int(0.95 * len(dataset["train"]))
    # dataset_num = int(0.95 * 30000)
    train_dataset = dataset["train"].select(range(0, dataset_num))
    eval_dataset = dataset["train"].select(range(dataset_num , dataset_num + 200))

    test_dataset = dataset["train"].select(range( dataset_num + 200, dataset_num + 400 ))

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    # eval_dataset = eval_dataset[:200]
    test_dataset = [format_data(sample) for sample in test_dataset]

    print(f'Examples :  \n data number : {len(train_dataset)}')

    for d in train_dataset[:1]:
        print(d)
        print('\n')
    print('===========================================')

    # 모델 및 프로세서 로드
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        # model_id, device_map=f"cuda:{local_rank}", torch_dtype='auto'
        model_id, device_map=f"cuda", torch_dtype='auto'
    )

    processor = AutoProcessor.from_pretrained(
         model_id,
        #  padding_side = "right",
    min_pixels=256*28*28,
    max_pixels=960*28*28
     )
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir="ex_models/qwen25-3b-Chart2Table_Vari_Query" , # Directory to save the model
        num_train_epochs=2,  # Number of training epochs
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation

        gradient_accumulation_steps=4,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        
        
        # Optimizer and scheduler settings
        optim="adafactor",  # Optimizer type
        # optim = 'paged_adamw_8bit',
        learning_rate=3e-5,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=100,  # Steps interval for logging
        eval_steps=500,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=10000,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        # fsdp="full_shard",         # FSDP 활성화
        # fsdp_min_num_params=1e8,     # 작은 모델은 FSDP 적용하지 않도록 임계값 설정
        
        # fp16=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        
        
        ddp_find_unused_parameters=False,    # DDP에서 사용되지 않는 파라미터를 탐지하지 않음
        dataloader_num_workers=1,            # DataLoader 멀티스레드 처리
        report_to=["none"],
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        local_rank=-1
        
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    # 학습
    trainer.train()

    # 모델 저장
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()