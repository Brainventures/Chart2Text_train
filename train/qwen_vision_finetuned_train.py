import torch
from torch.optim import AdamW
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor , AutoModelForCausalLM 
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import gc
import time
from transformers import TrainingArguments, Trainer ,EarlyStoppingCallback
# from peft import LoraConfig, get_peft_model , PeftModel

from trl import SFTConfig

# RTX Issue
import os

# 특정 워닝 무시
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# RTX Issue


# 데이터 전처리 함수 (변경 없음)
def format_data(sample):
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
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


# 데이터셋 로드
clear_memory()
# dataset_id = "HuggingFaceM4/ChartQA"


nia_path = '/workspace/data_moder/old_query_GPT_ALL.json'
# nia_path = '/workspace/data_moder/GPT_TABLE_DES_QA.json'
dataset = load_dataset('json',  data_files = nia_path )

# 데이터셋 분할
dataset_num = int(0.5 * len(dataset["train"]))
# dataset_num = int(0.95 * 30000)
train_dataset = dataset["train"].select(range(0, dataset_num))
eval_dataset = dataset["train"].select(range(dataset_num , dataset_num + 400))

test_dataset = dataset["train"].select(range( dataset_num + 400, dataset_num + 800 ))


train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
# eval_dataset = eval_dataset[:200]
test_dataset = [format_data(sample) for sample in test_dataset]

# print('Examples : \n')
# print(len(train_dataset))
# for d in train_dataset[:5]:
#     print(d)
#     print('\n')
# print('===========================================')





# 모델 및 프로세서 로드
# /workspace/Qwen/ex_models/qwen25-7b-Chart2Table
# model_id = "/workspace/Qwen/ex_models/qwen25-3b-Chart2Table_Vari_Query" # << 이게 방법론 다 적용된 것들. 
model_id = "/workspace/Qwen/ex_models/qwen25-3b-Chart2Table"
processor = AutoProcessor.from_pretrained(
     model_id,
     use_fast=True,
    #  padding_side = "right",
min_pixels=256*28*28,
max_pixels=960*28*28
 )

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_id, device_map="cuda", torch_dtype=torch.float16
# )
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # model_id, device_map=f"cuda:{local_rank}", torch_dtype='auto'
    model_id, device_map=f"cuda", torch_dtype=torch.bfloat16
)



# 데이터 Collator (변경 없음)
def collate_fn(examples):
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



# Configure training arguments
training_args = SFTConfig(
    output_dir="ex_models/qwen25-3b-ALL_no_vari_query" , # Directory to save the model
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation


    gradient_accumulation_steps=2,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    
    
    # Optimizer and scheduler settings
    # optim="adamw_torch",  # Optimizer type
    optim='adafactor',
    # optim = 'paged_adamw_8bit',
    learning_rate=3e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=100,  # Steps interval for logging
    eval_steps=5000,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20000,  # Steps interval for saving
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

    # Hub and reporting
    # push_to_hub=True,  # Whether to push model to Hugging Face Hub
    # report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
    local_rank=-1
    # local_rank = local_rank , 
    
)


# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

# 학습
trainer.train()

# 모델 저장
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
