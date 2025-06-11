import time 
import torch 
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="auto"
# )


checkpoint_path = "/workspace/Qwen/ex_models/qwen25-3b-instruct_ALL_vari_QA"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint_path, torch_dtype=torch.bfloat16, device_map="cuda"
)

# print(model)
# exit()


processor = AutoProcessor.from_pretrained(
     checkpoint_path,
     padding_side = "right",
min_pixels=256*28*28,
max_pixels=960*28*28
 )
model.eval()



torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)



# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8", min_pixels=min_pixels, max_pixels=max_pixels)



def format_data(image_dir):
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
                    # "text":'이 차트를 테이블로 변형하라.',
                    # 'text': '이 차트 이미지를 바탕으로 그림을 설명문을 작성해. ',
                    # 'text': '이 차트 이미지를 바탕으로 그림을 설명문을 작성하고, 설명한 내용을 바탕으로 요약문을 작성하라.',
                    "text" : '차트를 바탕으로 간단하게 설명해봐.'
                    # "text" : '차트를 보고 테이블로 번역해줘..'

                    # "text":'이 차트에서 가장 높은 수치가 무엇인가? 이에 대한 범례도 같이 말해라.',
                },
            ],
        },
        # {
        #     "role": "assistant",
        #     "content": [{"type": "text", "text": sample["label"][0]}],
        # },
    ]

image_dir = [
# '/data/Nia_data/blur_image_gpt/C_Source_294011_pie_standard.jpg',
# '/data/Nia_data/blur_image_gpt/C_Source_081570_pie_standard.jpg',
# '/data/Nia_data/blur_image_gpt/C_Source_267632_horizontal bar_standard.jpg',
# '/data/Nia_data/blur_image_gpt/C_Source_034032_vertical bar_accumulation.jpg',
# '/data/Nia_data/blur_image_gpt/C_Source_262379_horizontal bar_accumulation.jpg',
# '/workspace/Qwen/demo_image/demo_chart.png'
# ,'/workspace/Qwen/demo_image/micro_dust.png'
# ,'/workspace/Qwen/demo_image/covid_accumulated.png'
# ,'/workspace/Qwen/demo_image/graph04.png',
'/workspace/Qwen/demo_image/kospi.png'
    
]

t1 = time.time()
for i_d in image_dir:
    messages = format_data(i_d)
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")



    # print(inputs)
    # Inference: Generation of the output
    t11 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    # print(generated_ids)
    # Decode outputs
    # output_text = processor.batch_decode(
    #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    t12 = time.time()

    print(i_d.split('/')[-1],' is done. \t time :',t12-t11,'\n')
    print(f'{i_d}\'s OUTPUT :::::::::::')
    print(output_text[0])
t2 = time.time()

print('TOTAL TIME : ',t2-t1)
print('')