import torch 

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm 
# Re-importing libraries after the environment reset
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction
from rouge_score import rouge_scorer
import metrics as deplot_metric
import numpy as np 


import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Load the model in half-precision
model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# print(processor.tokenizer.padding_side)

# print(checkpoint_path)



def format_data(image_dir , query):
    system_message = """당신은 시각적 데이터(차트 이미지)를 분석하여 질문에 간결한 답변을 제공하는 비전 언어 모델입니다.
    차트 유형(예: 선 그래프, 막대 그래프, 원형 그래프), 색상, 레이블, 텍스트를 기반으로 질문에 답하세요.
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
                    "image": image_dir,
                },
                {
                    "type": "text",
                    "text":query,
                },
            ],
        },
        # {
        #     "role": "assistant",
        #     "content": [{"type": "text", "text": sample["label"][0]}],
        # },
    ]

def get_measure(refer, predict):
    # BLEU score 계산 함수
    def calculate_bleu(reference, prediction):
        smooth = SmoothingFunction().method1
        return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)

    # BLEU-RT score 계산 함수: 기존 BLEU 점수에 Brevity Penalty 조정을 추가
    def calculate_bleu_rt(reference, prediction):
        candidate = prediction.split()
        ref_tokens = reference.split()
        candidate_len = len(candidate)
        reference_len = len(ref_tokens)
        
        # 기본 BLEU score 계산 (smoothing 적용)
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], candidate, smoothing_function=smooth)
        
        # 단순화한 Brevity Penalty 적용:
        # 후보 길이가 기준보다 짧을 경우 패널티를 부여
        if candidate_len > reference_len:
            bp = 1.0
        else:
            bp = np.exp(1 - (reference_len / candidate_len))
        
        return bleu * bp

    # ROUGE score 계산 함수
    def calculate_rouge(reference, prediction):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores

    # F1 score 계산 (단어 단위)
    def calculate_f1(reference, prediction):
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        common = set(reference_tokens) & set(prediction_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(prediction_tokens)
        recall = len(common) / len(reference_tokens)
        return 2 * (precision * recall) / (precision + recall)
    
    bleu_score = calculate_bleu(refer, predict)
    bleu_rt = calculate_bleu_rt(refer, predict)
    rouge_scores = calculate_rouge(refer, predict)
    f1 = calculate_f1(refer, predict)

    return {'bleu': bleu_score,
            'bleu-rt': bleu_rt,
            'rouge': rouge_scores['rouge1'].fmeasure,
            'f1': f1}


nia_path = '/workspace/data_moder/GPT_TABLE.json'
# nia_path = '/workspace/data_moder/GPT_TABLE_DES_QA.json'
# nia_path = '/workspace/data_moder/gpt_ALL.json'


dataset = load_dataset('json',  data_files = nia_path )
# 데이터셋 분할
dataset_num = int(0.7 * len(dataset["train"]))
eval_num = int(0.05 * len(dataset["train"]))

train_dataset = dataset["train"].select(range(0, dataset_num))
eval_dataset = dataset["train"].select(range(dataset_num , dataset_num + eval_num))

test_dataset = dataset["train"].select(range( dataset_num + eval_num, dataset_num + eval_num*2 ))

print(test_dataset)

f1_lst = []
rouge_lst = []
bleu_lst = []
bleu_rt_lst = []
RMS_precision_lst = []
RMS_recall_lst = []
RMS_f1_lst = []
RNSS_lst = []
batch_size = 16

def evaluate_batch(batch):
    # 메시지와 입력 데이터 준비
    texts, image_inputs_list, video_inputs_list, references = [], [], [], []
    # for image_dir, query, label in zip(batch['image'], batch['query'], batch['label']):
    references.extend(batch['label'])  # 참조 데이터 저장
    #     # 메시지 포맷
    #     messages = format_data(image_dir, query)

    messages = [format_data(image_dir,query)for image_dir , query in zip(batch['image'],batch['query'])]
    

        # Text와 Vision 정보 생성
    # print(messages)

    inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    padding=True,
    return_tensors="pt"
    ).to(model.device, torch.float16)
    # 생성
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True)
    output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # print(output_texts)
    # exit()
    # # 배치로 생성된 출력 디코딩
    # output_texts = []
    # for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
    #     generated_ids_trimmed = out_ids[len(in_ids):]
    #     decoded = processor.decode(
    #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )
    #     output_texts.append(decoded)

    return output_texts, references

for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
    # 현재 배치

    batch = test_dataset[i : i + batch_size]
    batch_generated, batch_references = evaluate_batch(batch)
    
    # 메트릭 계산


    for idx, (generated, reference) in enumerate(zip(batch_generated, batch_references)):
        measured = get_measure(reference, generated)
        f1_lst.append(measured['f1'])
        rouge_lst.append(measured['rouge'])
        bleu_lst.append(measured['bleu'])
        bleu_rt_lst.append(measured['bleu-rt'])

        if nia_path.split('/')[-1] == 'GPT_TABLE.json':
        
            try:

                RMS = deplot_metric.table_datapoints_precision_recall([reference], [generated], text_theta=1.0)
                RNSS = deplot_metric.table_number_accuracy([reference], [generated])
            except:
                print(batch['image'][idx]    )
                print(reference)
                print(generated)
                continue
            RMS_precision_lst.append(RMS['table_datapoints_precision'])
            RMS_recall_lst.append(RMS['table_datapoints_recall'])
            RMS_f1_lst.append(RMS['table_datapoints_f1'])
            RNSS_lst.append(RNSS['numbers_match'])


print(f'TOTAL F1 : {sum(f1_lst)/len(f1_lst)}')
print(f'TOTAL ROUGE : {sum(rouge_lst)/len(rouge_lst)}')
print(f'TOTAL BLEU : {sum(bleu_lst)/len(bleu_lst)}')
print(f'TOTAL BLEU-RT : {sum(bleu_rt_lst)/len(bleu_rt_lst)}')

if nia_path.split('/')[-1] == 'GPT_TABLE.json':
    print(f'TOTAL RMS Precsion : {sum(RMS_precision_lst)/len(RMS_precision_lst)}')
    print(f'TOTAL RMS Recall : {sum(RMS_recall_lst)/len(RMS_recall_lst)}')
    print(f'TOTAL RMS F1 : {sum(RMS_f1_lst)/len(RMS_f1_lst)}')

    print(f'TOTAL RNSS : {sum(RNSS_lst)/len(RNSS_lst)}')

