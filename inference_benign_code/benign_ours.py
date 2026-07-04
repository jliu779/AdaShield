import sys, os
import shutil

import argparse

import torch
import json
from tqdm import tqdm

# llava
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

## cogvlm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# minigptv4

import torch.backends.cudnn as cudnn
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")

import clip
from PIL import Image
import imageio
import random
import pandas as pd
import glob
import datetime

# 新增 MLLM 模型路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LLAVA15_PATH, QWEN25VL_PATH, INTERNVL35_PATH,
                    INTERNVL3_PATH, QWEN3VL_PATH, GLM41V_PATH)


def _load_hf_vlm(model_path, device="cuda"):
    """统一加载 HuggingFace VLM 模型 —— 依次尝试多种 auto class"""
    model = None
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
    except (ValueError, ImportError, KeyError):
        pass
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True,
            ).eval()
        except ValueError:
            pass
    if model is None:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def _hf_vlm_generate(model, processor, query, image_path,
                     max_new_tokens=2048, temperature=0.2, top_p=0.7):
    """统一 HuggingFace VLM 推理"""
    image = Image.open(image_path).convert('RGB')
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": query}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text, images=image, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return response


def BenignTest_HF(args):
    """统一的 HF VLM 良性评估函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    answers_file = os.path.expanduser(args.result_path)

    with open(args.query_file_path) as f:
        query_list = json.load(f)

    model, processor = _load_hf_vlm(args.model_path)

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    total = len(query_list)
    print("Total files:", total)
    answers = {}

    print("sample-wise defense prompt retraival")
    print("start to get defense example pool!")
    defense_prompt_pool, embedding_pool, _ = get_all_defense_example_pool(
        args, clip_model, clip_preprocess, args.table_dir, -1)
    print(" END !")

    with tqdm(total=total) as par:
        for i in query_list:
            image_file = query_list[i]["imagename"]
            query = query_list[i]["question"]
            full_image_path = os.path.join(args.image_folder, image_file)
            sample_embedding = get_clip_embedding(
                clip_model, clip_preprocess, full_image_path, query)

            defense_query, best_similarity = retrival_defense_prompt(
                defense_prompt_pool,
                retrival_type="sample-wise",
                sample_embedding=sample_embedding,
                embedding_pool=embedding_pool)

            if best_similarity > args.beta:
                query = query + defense_query + query

            response = _hf_vlm_generate(
                model, processor, query, full_image_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p)

            answers[i] = response
            par.update(1)

    ans_file.write(json.dumps(answers))
    ans_file.flush()
    ans_file.close()


def read_file_to_string(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content


def load_image_paths_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") \
                or filename.endswith(".JPEG"):  # Add more file types if needed
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
    return image_paths


def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset


@torch.no_grad()
def get_clip_embedding(clip_model, clip_preprocess, image_file, query):
    image = clip_preprocess(load_image(image_file)).unsqueeze(0).cuda()
    text = clip.tokenize(query).cuda()
    image_feature = clip_model.encode_image(image)
    text_feature = clip_model.encode_text(text)

    embedding = torch.cat((image_feature,text_feature),dim=-1) # (1,1024)
    return torch.nn.functional.normalize(embedding, dim=-1)

def retrival_defense_prompt(prompt_pool, retrival_type= "scenario-wise", sample_embedding=None, embedding_pool=None):
    if retrival_type == "scenario-wise": # 基于场景去随机retrival defense
        index = random.randint(0, len(prompt_pool)-1)
        final_defense_prompt = prompt_pool[index]
        best_similarity = 0

    elif retrival_type == "sample-wise": 
        assert sample_embedding is not None 
        assert embedding_pool is not None
        similarity = (sample_embedding.float() @ embedding_pool.float().t()).squeeze() 
        best_indice_list = torch.nonzero(similarity == similarity.max()) 
        if best_indice_list.shape[0] ==1:
            random_best_index = best_indice_list
        else:
            random_best_index = random.choice(best_indice_list)
        final_defense_prompt = prompt_pool[random_best_index]
        best_similarity = similarity[random_best_index].item()
    else:
        raise ValueError("retrival_type should be either'scenario-wise' or'sample-wise'.")
    return final_defense_prompt, best_similarity 


def get_all_defense_example_pool(args, clip_model, clip_preprocess, table_dir ,defense_number=-1):
    scenerio_list = [ "01-Illegal_Activitiy",
            "02-HateSpeech", "03-Malware_Generation", "04-Physical_Harm",
            "05-EconomicHarm", "06-Fraud", "07-Sex", "08-Political_Lobbying", 
            "09-Privacy_Violence", "10-Legal_Opinion", "11-Financial_Advice", "12-Health_Consultation", "13-Gov_Decision" ]
    embedding_pool = None
    image_pool  = []
    final_defense_pool = []
    for scene in scenerio_list:
        print(os.path.join(table_dir, scene, "wandb", args.wandb_subdir ,"files/final_table.csv"))
        try:
            table_file_list = glob.glob(os.path.join(table_dir, scene, "wandb", args.wandb_subdir ,"files/final_table.csv"))
            if len(table_file_list)==0:
                print("No table file found")
            
            table_name = table_file_list[0]
            full_df = pd.read_csv(table_name)
            if defense_number <0 :
                defense_number =len(full_df)
                temp_df = full_df
            else:
                defense_number = min(len(full_df), defense_number)
                temp_df = full_df.sample(frac=1).reset_index(drop=True).head(defense_number)

            success_df = temp_df[temp_df["final_judge_scores"]==1]

            temp_image_pool =  success_df["image"].tolist() # image path
            temp_defense_pool = success_df["defense_prompt_list"].tolist()
            temp_query_pool =  success_df["query"].tolist()
        
            image_pool += temp_image_pool
            final_defense_pool += temp_defense_pool
        
            for i in tqdm(range(len(temp_image_pool))):
                image_file = temp_image_pool[i]
                query = temp_query_pool[i]
                embedding = get_clip_embedding(clip_model, clip_preprocess, image_file, query)
                if embedding_pool is None:
                    embedding_pool = torch.zeros(0,embedding.shape[1]).to(embedding.device)
                embedding_pool = torch.cat((embedding_pool,embedding),dim=0)
        except Exception as e:
            break
    return final_defense_pool, embedding_pool, image_pool 


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def get_test_sample_list(args):
    split_file = open(args.data_split_file, 'r', encoding='utf-8')
    split_list = []
    for line in split_file.readlines():
        dic = json.loads(line)
        split_list.append(dic)
        
    for item in split_list:
        if item.get(args.scenario):
            test_img_list = item[args.scenario]['test']
            break
        else:
            pass

    return test_img_list


def VQA(chat, image_file, prompt):
    image_file = image_file.convert('RGB')
    prompt = prompt[:2000]
    
    
    CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
    )

    # ask
    chat_state = CONV_VISION.copy()
    chatbot = []
    img_list = []
    llm_message = chat.upload_img(image_file, chat_state, img_list)

    chat.ask(prompt, chat_state)

    chatbot = chatbot + [[prompt, None]]

    # answer
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)

    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              temperature=0.2,
                              max_new_tokens=500,
                              max_length=2048)[0]

    return llm_message

def MiniGPT4BenignTest(chat, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    answers_file = os.path.expanduser(args.result_path)


    with open(args.query_file_path) as f:
        query_list = json.load(f)


    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    total = len(query_list)
    print("Total files:",total)
    answers = {}
        
    print("sample-wise defense prompt retraival")
    print("start to get defense example pool!")
    defense_prompt_pool, embedding_pool, _ = get_all_defense_example_pool(args, clip_model, clip_preprocess, args.table_dir, -1)  #get_defense_example_pool(args, conv_mode, tokenizer, model, image_processor, table_file, defense_number=args.defense_number)
    print(" END !")


    with tqdm(total=total) as par:
        for i in query_list:
            image_file = query_list[i]["imagename"]
            query = query_list[i]["question"]
            full_image_path = os.path.join(args.image_folder, image_file)
            sample_embedding = get_clip_embedding(clip_model, clip_preprocess, full_image_path, query) #(args, conv_mode, tokenizer, model, image_processor, full_image_path, query)
            
            defense_query, best_simiarity = retrival_defense_prompt(defense_prompt_pool, retrival_type= "sample-wise", sample_embedding=sample_embedding, embedding_pool=embedding_pool)
            if best_simiarity > 7.1:
                query  = query + defense_query + query
            
            image = Image.open(full_image_path).convert('RGB')
            with torch.no_grad():
                respone = VQA(chat, image, query)
            answers[i] = respone
            par.update(1)
    print(answers)
    ans_file.write(json.dumps(answers))
    ans_file.flush()
    ans_file.close()

def LLaVABenignTest(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    answers_file = os.path.expanduser(args.result_path)

    with open(args.query_file_path) as f:
        query_list = json.load(f)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()
    
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    total = len(query_list)
    print("Total files:",total)
    answers = {}
    
    print("sample-wise defense prompt retraival")
    print("start to get defense example pool!")
    defense_prompt_pool, embedding_pool, _ = get_all_defense_example_pool(args, clip_model, clip_preprocess, args.table_dir, -1)  #get_defense_example_pool(args, conv_mode, tokenizer, model, image_processor, table_file, defense_number=args.defense_number)
    print(" END !")

    with tqdm(total=total) as par:
        for i in query_list:

            image_file = query_list[i]["imagename"]
            qs = query_list[i]["question"]
            full_image_path = os.path.join(args.image_folder, image_file)
            sample_embedding = get_clip_embedding(clip_model, clip_preprocess, full_image_path, qs) 
            
            defense_query, best_simiarity = retrival_defense_prompt(defense_prompt_pool, retrival_type= "sample-wise", sample_embedding=sample_embedding, embedding_pool=embedding_pool)
            
            if best_simiarity > args.beta :
                qs = qs + defense_query + qs

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is None or conv_mode == args.conv_mode:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


            image = Image.open(full_image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                print('stop str')
                outputs = outputs[:-len(stop_str)]
            answers[i] = outputs
            par.update(1)

    ans_file.write(json.dumps(answers))
    ans_file.flush()
    ans_file.close()

def CogVLMBenignTest(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    answers_file = os.path.expanduser(args.result_path)

    with open(args.query_file_path) as f:
        query_list = json.load(f)

    # Model
    tokenizer = LlamaTokenizer.from_pretrained(args.llm_path) #
    model_name = os.path.basename(args.llm_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True)

    model = load_checkpoint_and_dispatch(
                    model,
                    args.model_path,
                    device_map="auto",
                    no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
        
    model = model.eval()
    
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    total = len(query_list)
    print("Total files:",total)
    answers = {}
    
    print("sample-wise defense prompt retraival")
    print("start to get defense example pool!")
    defense_prompt_pool, embedding_pool, _ = get_all_defense_example_pool(args, clip_model, clip_preprocess, args.table_dir, -1)  
    print(" END !")

    with tqdm(total=total) as par:
        for i in query_list:
            image_file = query_list[i]["imagename"]
            query = query_list[i]["question"]
            
            full_image_path = os.path.join(args.image_folder, image_file)
            sample_embedding = get_clip_embedding(clip_model, clip_preprocess, full_image_path, query) 
            
            defense_query, best_simiarity = retrival_defense_prompt(defense_prompt_pool, retrival_type= "sample-wise", sample_embedding=sample_embedding, embedding_pool=embedding_pool)
            if best_simiarity > 7.1:
                query  = query + defense_query + query
            
            image = Image.open(full_image_path).convert('RGB')
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

            input_ids = inputs['input_ids'].unsqueeze(0).to('cuda')

            with torch.no_grad():
                outputs = model.generate(
                            input_ids = input_ids,
                            images = [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                            token_type_ids = inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                            attention_mask = inputs['attention_mask'].unsqueeze(0).to('cuda'),
                            do_sample = True,
                            temperature = args.temperature,
                            top_p = args.top_p,
                            top_k = args.top_k,
                            max_length = args.max_length)

                outputs = outputs[:, input_ids.shape[1]:]
                respone = tokenizer.decode(outputs[0])
            answers[i] = respone
            par.update(1)

    ans_file.write(json.dumps(answers))
    ans_file.flush()
    ans_file.close()

def main(args):
    if args.target_model == '':
        cfg = Config(args)
        device = 'cuda:{}'.format(args.gpu_id)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        bounding_box_size = 100

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        model = model.eval()



        chat = Chat(model, vis_processor, device=device)
        MiniGPT4BenignTest(chat,args)
    elif args.target_model == "llava":
        LLaVABenignTest(args)

    elif args.target_model == "cogvlm":
        CogVLMBenignTest(args)

    elif args.target_model in ("llava-1.5", "qwen2.5-vl", "internvl3.5",
                               "internvl3", "qwen3-vl", "glm-4.1v"):
        BenignTest_HF(args)

    else:
        print("please check your model name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment arguments for mm-vet dataset.")

    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--table_dir", type=str, default="wandb/llava")
    parser.add_argument("--wandb_subdir", type=str, default="latest")   

    parser.add_argument("--image_folder", type=str, default="data/mm-vet/images")
    parser.add_argument("--query_file_path", type=str, default="data/mm-vet/mm-vet.json")

    parser.add_argument("--result_path", type=str, default="results/benign/lava-ours.jsonl")


    parser.add_argument(
        "--target-model",
        default = "llava",
        help = "Name of target vision-language model.",
        choices=["cogvlm", "minigptv2", "llava",
                 "llava-1.5", "qwen2.5-vl", "internvl3.5", "internvl3", "qwen3-vl", "glm-4.1v"]
    )

    args = parser.parse_args()

    ############################### specific model config ###############################
    if args.target_model == "llava":
        args.model_path="modellib/llava-v1.5-13b"
        args.model_base=None
        args.conv_mode="vicuna_v1"
        args.sep=","
        args.temperature=0.2
        args.top_p=0.7
        args.num_beams=1
        args.max_new_tokens=3000

    elif args.target_model == "minigptv2":
        args.cfg_path="eval_config/minigptv2_eval.yaml"
        args.gpu_id=0
        args.options = None

    elif args.target_model == "cogvlm":
        args.model_path="modellib/cogvlm-chat-hf"
        args.llm_path="modellib/vicuna-7b-v1.5"
        args.model_base=None
        args.conv_mode=None
        args.sep=","
        args.temperature=0.8
        args.top_p=0.4
        args.top_k=1
        args.num_beams=1
        args.max_new_tokens=512
        args.max_length=3000

    elif args.target_model == "llava-1.5":
        args.model_path=LLAVA15_PATH
        args.temperature=0.2
        args.top_p=0.7
        args.max_new_tokens=2048

    elif args.target_model == "qwen2.5-vl":
        args.model_path=QWEN25VL_PATH
        args.temperature=0.2
        args.top_p=0.7
        args.max_new_tokens=2048

    elif args.target_model == "internvl3.5":
        args.model_path=INTERNVL35_PATH
        args.temperature=0.2
        args.top_p=0.7
        args.max_new_tokens=2048

    elif args.target_model == "internvl3":
        args.model_path=INTERNVL3_PATH
        args.temperature=0.2
        args.top_p=0.7
        args.max_new_tokens=2048

    elif args.target_model == "qwen3-vl":
        args.model_path=QWEN3VL_PATH
        args.temperature=0.2
        args.top_p=0.7
        args.max_new_tokens=2048

    elif args.target_model == "glm-4.1v":
        args.model_path=GLM41V_PATH
        args.temperature=0.2
        args.top_p=0.7
        args.max_new_tokens=2048

    main(args)