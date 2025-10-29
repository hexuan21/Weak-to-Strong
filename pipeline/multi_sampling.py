import asyncio
import logging
import os
import json
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import random
import transformers
import torch
import tiktoken
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from zeno_build.models import lm_config
import replicate

from utils_async import *


async def _async_output_DEEPBRICKS_GPT(context_list,model_config,model_name,logger,num_samples):
    max_tokens=8000
    if "gpt-3.5" in model_name:
        max_tokens=4096
    raw_outputs=[]
    for context_sublist in [context_list[i:i + CONTEXT_SUBLIST_MAX_LEN] for i in range(0, len(context_list), CONTEXT_SUBLIST_MAX_LEN)]:
        raw_outputs.extend(await generate_from_openai_chat_completion(
        full_contexts = context_sublist,
        model_config = model_config, 
        max_tokens=max_tokens, 
        temperature=1.0,
        logger=logger, 
        num_samples=num_samples))
    
    return raw_outputs


async def _async_output_DEEPBRICKS_OTHER(context_list,model_config,model_name,logger,num_samples):
    raw_outputs=[]
    for _ in range(num_samples):
        max_tokens=8000
        if "gpt-3.5" in model_name:
            max_tokens=4096
        curr_outputs=[]
        for context_sublist in [context_list[i:i + CONTEXT_SUBLIST_MAX_LEN] for i in range(0, len(context_list), CONTEXT_SUBLIST_MAX_LEN)]:
            curr_outputs.extend( await generate_from_openai_chat_completion(
            full_contexts = context_sublist,model_config = model_config, max_tokens=max_tokens, logger=logger,num_samples=1))
        if num_samples!=1:
            if raw_outputs == []:
                raw_outputs=[[curr_output] for curr_output in curr_outputs]
            else:
                for i, curr_output in enumerate(curr_outputs):
                    raw_outputs[i].append(curr_output)
        else:
            raw_outputs=[curr_output for curr_output in curr_outputs]

    return raw_outputs

async def _async_output_DEEPINFRA(context_list,model_config,logger,num_samples):
    max_tokens=4096
    raw_outputs=[]
    for context_sublist in [context_list[i:i + CONTEXT_SUBLIST_MAX_LEN] for i in range(0, len(context_list), CONTEXT_SUBLIST_MAX_LEN)]:
        raw_outputs.extend(await generate_from_openai_chat_completion(
        full_contexts = context_sublist,
        model_config = model_config, 
        max_tokens=max_tokens, 
        temperature=1.0,
        logger=logger, 
        num_samples=num_samples))
    
    return raw_outputs


def _output_DEEPINFRA_and_DEEPSEEK(context_list,model_name,num_samples,logger,):
    raw_outputs=[]
    for context in tqdm(context_list):
        if num_samples!=1:
            samples=[]
            for _ in range(num_samples):
                response = client.chat.completions.create(
                    model=model_name,
                        messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": context["messages"][0]["content"]},
                    ],
                    stream=False,
                )
                if logger is not None:
                    logger.info(f"\n{response.usage}")
                else:
                    print(f"\n{response.usage}")
                samples.append(response.choices[0].message.content)
            raw_outputs.append(samples)
        else:
            response = client.chat.completions.create(
                    model=model_name,
                        messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": context["messages"][0]["content"]},
                    ],
                    stream=False,
                )
            raw_outputs.append(response.choices[0].message.content)
    return raw_outputs


def _output_REPLICATE(context_list,model_name,num_samples,logger):
    raw_outputs=[]
    for context in tqdm(context_list):
        if num_samples!=1:
            samples=[]
            for _ in range(num_samples):
                text=context["messages"][0]["content"]       
                output = replicate.run(
                    model_name,
                    input={
                        "prompt": text,
                        "max_length": 4000
                    }
                )
                sample=""
                for item in output:
                    sample+=f"{item}"
                samples.append(sample)
            raw_outputs.append(samples)
        else:
            text=context["messages"][0]["content"]       
            output = replicate.run(
                model_name,
                input={
                    "prompt": text,
                    "max_length": 4000
                }
            )
            sample=""
            for item in output:
                sample+=f"{item}"
            raw_outputs.append(sample)
    return raw_outputs


    
def _output_HF(context_list,model_name,num_samples):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=HF_TOKEN,
    )
    raw_outputs=[]
    for context in context_list:
        samples=[]
        for _ in range(num_samples):
            text = context["messages"][0]["content"]       
            output = pipeline(
                text,
                max_new_tokens=4096,
            )
            
        samples.append(output[0]["generated_text"].strip(NOISE_CHARS))
        raw_outputs.append(samples)
    return raw_outputs


def load_task_truth_v1(data,full_or_sub,):
    if full_or_sub =="full":
        task_list=[item["original"]["task"] for item in data]
        pos_ans_list=[item["original"]["pos_ans"] for item in data]
    elif full_or_sub == "sub":
        num_subtask_list=[len([key for key in item.keys() if "sub" in key]) for item in data]
        task_list=[]
        pos_ans_list=[]
        for full_idx,item in enumerate(data):
            for sub_idx in range(num_subtask_list[full_idx]):
                task_list.append(item[f"subtask_{sub_idx+1}"]["task"])
                pos_ans_list.append(item[f"subtask_{sub_idx+1}"]["pos_ans"])
    else:
        print("var 'full_or_sub' must be 'full' or 'sub'")
        exit()
    
    return task_list,pos_ans_list
    

def load_task_truth_v2(data,full_or_sub,):
    if len(data)==0:
        return [],[],[]
    if "ori_id" in data[0] and "task" in data[0].keys() and "ground_truth" in data[0].keys():
        ori_id_list=[item["ori_id"] for item in data]
        task_list=[item["task"] for item in data]
        pos_ans_list=[item["ground_truth"] for item in data]

    else:
        print("key error")
        exit()
    
    return ori_id_list,task_list,pos_ans_list


async def multi_sampling(prompt,model_config,model_name,num_samples,task_list,pos_ans_list,out_file,logger,ori_id_list=None):
    context_list=[]
    for task,pos_ans in zip(task_list,pos_ans_list):
        pos_token_len=len(encoding.encode(pos_ans))
        user_input=prompt+"\n### Problem: \n"+task+f"Note that your output tokens SHOULE NOT be more than {int(1.5*pos_token_len)} and NOT be less than {int(0.5*pos_token_len)}"
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    
    raw_outputs=[]
    if model_name in DEEP_BRICKS_GPT_MODEL_LIST:
        raw_outputs = await _async_output_DEEPBRICKS_GPT(
            context_list=context_list,
            model_config = model_config, 
            model_name=model_name,
            logger=logger,
            num_samples=num_samples)
        
    elif model_name in DEEP_BRICKS_OTHER_MODEL_LIST:
        raw_outputs = await _async_output_DEEPBRICKS_OTHER(
            context_list=context_list,
            model_config = model_config, 
            model_name=model_name,
            logger=logger,
            num_samples=num_samples)
    
    elif model_name in DEEP_INFRA_MODEL_LIST:
        raw_outputs = await _async_output_DEEPINFRA(
            context_list=context_list,
            model_config = model_config, 
            logger=logger,
            num_samples=num_samples)
    
    elif model_name in DEEP_SEEK_MODEL_LIST:
        raw_outputs = _output_DEEPINFRA_and_DEEPSEEK(context_list=context_list,model_name=model_name,num_samples=num_samples,logger=None)
        
    elif model_name in REPLICATE_MODEL_LIST:
        raw_outputs = _output_REPLICATE(context_list=context_list,model_name=model_name,num_samples=num_samples,logger=None)
    
    else:
        print("model not supported")
        exit()

    out_items=[]
    for idx,(task,ground_truth,sample) in enumerate(zip(task_list,pos_ans_list,raw_outputs)):
        # logger.info(f"======================================== {idx}-th output ========================================")
        # logger.info(sample)
        
        out_item={}
        if ori_id_list is not None:
            out_item["ori_id"]=ori_id_list[idx]
        out_item["task"]=task
        out_item["ground_truth"]=ground_truth
        start_sign="Solution"
        if num_samples!=1:
            for idx,one_sample in enumerate(sample):
                
                if start_sign in one_sample:
                    one_sample=one_sample.replace(start_sign,"")
                out_item[f"sample{idx+1}"]=one_sample.strip(NOISE_CHARS)
        else:
            if start_sign in sample:
                sample=sample.replace(start_sign,"")
            out_item[f"sample1"]=sample.strip(NOISE_CHARS)
        out_items.append(out_item)

    os.makedirs(os.path.dirname(out_file),exist_ok=True)
    with open(out_file,"w") as file:
        json.dump(out_items,file,indent=4)




if __name__ =="__main__":
    s_t=time.time()
    NUM_SAMPLES=3
    CONTEXT_SUBLIST_MAX_LEN=400
    
    templates=json.load(open("./const/prompt_template.json","r",encoding="utf-8"))
    gen_ans_prompt=templates["multi_sampling_gen_ans_0909"]
    
    API_KEYS=json.load(open("./const/api_key.json","r"))
    HF_TOKEN=API_KEYS["HF_TOKEN_v2"]
    
    DEEP_BRICKS_GPT_MODEL_LIST=["gpt-4o-2024-08-06","gpt-4o","gpt-4o-mini","gpt-3.5-turbo","gpt-3.5-turbo-instruct",]
    DEEP_BRICKS_OTHER_MODEL_LIST=["claude-3.5-sonnet","llama-3.1-405b","llama-3.1-70b","llama-3-70b"]
    DEEP_INFRA_MODEL_LIST=["meta-llama/Meta-Llama-3.1-70B-Instruct","meta-llama/Meta-Llama-3-70B-Instruct"]
    DEEP_SEEK_MODEL_LIST=["deepseek-coder","deepseek-chat"]
    REPLICATE_MODEL_LIST=["meta/meta-llama-3-70b-instruct"]
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    
    
    # dataset="numina_syn_amc"
    # BATCH_START=22
    # BATCH_END=25
    # api_key_idx=1
    # used_size="all"
    # trial=0
    
    # full_or_sub="sub"
    # # model_name="gpt-4o-mini"
    # model_name="gpt-3.5-turbo"
    # # model_name="llama-3-70b"
    
    # if model_name in DEEP_BRICKS_GPT_MODEL_LIST or model_name in DEEP_BRICKS_OTHER_MODEL_LIST:
    #     os.environ["OPENAI_API_KEY"]=API_KEYS[f"OpenAI_API_KEYd{api_key_idx}"]
    #     os.environ["OPENAI_ORG"]=""
    #     os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepBricks_BASE_URL"]
    # elif model_name in DEEP_SEEK_MODEL_LIST:
    #     os.environ["OPENAI_API_KEY"]=API_KEYS["DeepSeek_API_KEY"]
    #     os.environ["OPENAI_ORG"]=""
    #     os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepSeek_BASE_URL"]
    # else:
    #     print("model not supported")
    #     exit()
        
    # ASYNC_MODEL_CONFIG = lm_config.LMConfig(provider="openai_chat", model=model_name)
    # client=None
    # if model_name in DEEP_SEEK_MODEL_LIST:
    #     client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    
    
    # for batch_idx in range(BATCH_START,BATCH_END+1):
    #     src_file=f"./res/{dataset}/good/{dataset}_{batch_idx}_good.json"
    #     out_dir=f'./res_multi_sample/{dataset}/{full_or_sub}'
    #     out_file=f'{out_dir}/{dataset}_{full_or_sub}_{batch_idx}.json'
        
    #     if trial==1:
    #         out_dir=f'./res_multi_sample/trial_{dataset}/{full_or_sub}'
    #         if "/" in model_name:
    #             simple_model_name=model_name.replace("/","-")
    #         else:
    #             simple_model_name=model_name
    #         out_file=f'{out_dir}/{dataset}_{full_or_sub}_{batch_idx}_{simple_model_name}.json'
        
    #     src_data=json.load(open(src_file,"r"))
    #     if type(used_size) is int:
    #         src_data=src_data[:used_size]

    #     task_list=[]
    #     pos_ans_list=[]
    #     task_list,pos_ans_list=load_task_truth_v1(data=src_data,full_or_sub=full_or_sub)
        
    #     os.makedirs(os.path.dirname(out_file),exist_ok=True)
    #     logger_file=f"./logs_multi_sample/{dataset}_{full_or_sub}/{dataset}_{full_or_sub}_{batch_idx}.log"
    #     logger=set_logger(logger_file)
    #     asyncio.run(multi_sampling(prompt=gen_ans_prompt,model_config=ASYNC_MODEL_CONFIG,
    #                             num_samples=NUM_SAMPLES,
    #                             task_list=task_list,pos_ans_list=pos_ans_list,
    #                             out_file=out_file,
    #                             logger=logger,))

        
    #     print(f" =============================== time: {time.time()-s_t} =============================== ")
    
    
    NUM_SAMPLES=1
    
    api_key_idx=1
    BATCH_START=0
    BATCH_END=21
    used_size="all"

    # model_name="gpt-4o-mini"
    # model_name="gpt-3.5-turbo"
    # model_name="gpt-3.5-turbo-instruct"
    # model_name="llama-3-70b"
    # model_name="llama-3-70b"
    # model_name="llama-3.1-405b"
    # model_name="meta-llama/Meta-Llama-3-70B-Instruct"
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    if model_name in DEEP_BRICKS_GPT_MODEL_LIST or model_name in DEEP_BRICKS_OTHER_MODEL_LIST:
        os.environ["OPENAI_API_KEY"]=API_KEYS[f"OpenAI_API_KEYd{api_key_idx}"]
        os.environ["OPENAI_ORG"]=""
        os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepBricks_BASE_URL"]
    elif model_name in DEEP_SEEK_MODEL_LIST:
        os.environ["OPENAI_API_KEY"]=API_KEYS["DeepSeek_API_KEY"]
        os.environ["OPENAI_ORG"]=""
        os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepSeek_BASE_URL"]
    elif model_name in DEEP_INFRA_MODEL_LIST:
        os.environ["OPENAI_API_KEY"]=API_KEYS[f"DeepInfra_API_KEY{api_key_idx}"]
        os.environ["OPENAI_ORG"]=""
        os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepInfra_BASE_URL"]
    elif model_name in REPLICATE_MODEL_LIST:
        os.environ["REPLICATE_API_TOKEN"]=API_KEYS["REPLICATE_API_TOKEN"]
    else:
        print("model not supported")
        exit()
    
    
    ASYNC_MODEL_CONFIG = lm_config.LMConfig(provider="openai_chat", model=model_name)
    client=None
    if model_name in DEEP_SEEK_MODEL_LIST or model_name in DEEP_INFRA_MODEL_LIST:
        client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    
    for batch_idx in range(BATCH_START,BATCH_END+1):
        # src_file=f"./weaker_sampling/command-r-2024-08/math_sft_full_command-r-2024-08_checked_{batch_idx}.json"
        src_file=f"./weaker_sampling/Meta-Llama-3.1-70B-Instruct_bad/math_sft_full_Meta-Llama-3.1-70B-Instruct_to_check_{batch_idx}.json"
        if "/" in model_name:
            simple_model_name=model_name.split("/")[1]
        else:
            simple_model_name=model_name

        # out_file=f'./weaker_sampling/{simple_model_name}/math_sft_full_{simple_model_name}_to_check_{batch_idx}.json'
        out_file=f'./weaker_sampling/{simple_model_name}_bad2/math_sft_full_{simple_model_name}_to_check_{batch_idx}.json'
        os.makedirs(os.path.dirname(out_file),exist_ok=True)
        
        src_data=json.load(open(src_file,"r"))
        if type(used_size) is int:
            src_data=src_data[:used_size]
        ori_id_list,task_list,pos_ans_list=[],[],[]
        ori_id_list,task_list,pos_ans_list=load_task_truth_v2(data=src_data,full_or_sub="full")
        if len(task_list)==0:
            continue
        print(len(task_list))
        
        # logger_file=f"logs_multi_sample/weak_sampling/{model_name}_{batch_idx}.log"
        # logger=set_logger(logger_file=logger_file)
        logger=None
        os.makedirs(os.path.dirname(out_file),exist_ok=True)
        asyncio.run(multi_sampling(prompt=gen_ans_prompt,model_config=ASYNC_MODEL_CONFIG,
                                    model_name=model_name,
                                num_samples=NUM_SAMPLES,
                                task_list=task_list,pos_ans_list=pos_ans_list,
                                out_file=out_file,logger=logger,ori_id_list=ori_id_list))

            
        print(f" =============================== time: {time.time()-s_t} =============================== ")
    
    
    
    