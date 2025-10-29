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


def load_full_task(data,):
    if "ori_id" in data[0] and "task" in data[0]['original'].keys():
        ori_id_list=[item["ori_id"] for item in data]
        task_list=[item['original']["task"] for item in data]
        pos_ans_list=[item['original']["pos_ans"] for item in data]
        neg_ans_list=[item['original']["neg_ans"] for item in data]

    else:
        print("key error")
        exit()
    
    return ori_id_list,task_list,pos_ans_list,neg_ans_list


async def paraphrase(prompt,model_config,model_name,num_samples,out_file,logger,task_list,pos_ans_list,neg_ans_list,ori_id_list=None):
    ###### =============================== Paraphrase Problem ======================================================
    context_list=[]
    for task,_,_ in zip(task_list,pos_ans_list,neg_ans_list):
        user_input=prompt+"\n### Problem: \n"+task
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    raw_outputs = await _async_output_DEEPBRICKS_GPT(
        context_list=context_list,model_config = model_config, model_name=model_name,logger=logger,num_samples=num_samples)

    new_task_list=[]
    for idx,raw_output in enumerate(raw_outputs):
        out_item={}
        if ori_id_list is not None:
            out_item["ori_id"]=ori_id_list[idx]
        
        sign="Paraphrased version"
        if sign in raw_output:
            new_task=raw_output.split(sign)[1].strip(NOISE_CHARS)
        else:
            new_task=""
        out_item["task"]=new_task
        new_task_list.append(out_item)



    ###### ========================== Paraphrase Correct Solution ==========================================
    context_list=[]
    for _,pos_ans,_ in zip(task_list,pos_ans_list,neg_ans_list):
        user_input=prompt+"\n### Solution: \n"+pos_ans
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    raw_outputs = await _async_output_DEEPBRICKS_GPT(
        context_list=context_list,model_config = model_config, model_name=model_name,logger=logger,num_samples=num_samples)

    new_sol_1_list=[]
    for idx,raw_output in enumerate(raw_outputs):
        out_item={}
        if ori_id_list is not None:
            out_item["ori_id"]=ori_id_list[idx]
        
        sign="Paraphrased version"
        if sign in raw_output:
            new_sol=raw_output.split(sign)[1].strip(NOISE_CHARS)
        else:
            new_sol=""
        out_item["pos_ans"]=new_sol
        new_sol_1_list.append(out_item)
    
    
    
    ###### =============================== Paraphrase Incorrect Solution ======================================================
    context_list=[]
    for _,_,neg_ans in zip(task_list,pos_ans_list,neg_ans_list):
        user_input=prompt+"\n### Solution: \n"+neg_ans
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    raw_outputs = await _async_output_DEEPBRICKS_GPT(
        context_list=context_list,model_config = model_config, model_name=model_name,logger=logger,num_samples=num_samples)

    new_sol_2_list=[]
    for idx,raw_output in enumerate(raw_outputs):
        out_item={}
        if ori_id_list is not None:
            out_item["ori_id"]=ori_id_list[idx]
        
        sign="Paraphrased version"
        if sign in raw_output:
            new_sol=raw_output.split(sign)[1].strip(NOISE_CHARS)
        else:
            new_sol=""
        out_item["neg_ans"]=new_sol
        new_sol_2_list.append(out_item)
    
    print(len(new_task_list),len(new_sol_1_list),len(new_sol_2_list))
    
    out_items=[]
    for x,y,z in zip(new_task_list,new_sol_1_list,new_sol_2_list):
        assert x["ori_id"]==y["ori_id"] and y["ori_id"]==z["ori_id"]
        ori_id=x["ori_id"]
        new_task=x["task"]
        new_pos_ans=y["pos_ans"]
        new_neg_ans=z["neg_ans"]
        out_items.append({
            "ori_id":ori_id,
            "task":new_task,
            "pos_ans":new_pos_ans,
            "neg_ans":new_neg_ans  
        })
        
    os.makedirs(os.path.dirname(out_file),exist_ok=True)
    with open(out_file,"w") as file:
        json.dump(out_items,file,indent=4)




if __name__ =="__main__":
    s_t=time.time()
    CONTEXT_SUBLIST_MAX_LEN=400
    
    templates=json.load(open("./const/prompt_template.json","r",encoding="utf-8"))
    paraphrase_prompt=templates["paraphrase_0926"]
    
    API_KEYS=json.load(open("./const/api_key.json","r"))    
    NUM_SAMPLES=1
    
    api_key_idx=1
    subset="numina_syn_math"
    
    BATCH_START=22
    BATCH_END=22
    used_size="all"
    
    model_name="gpt-4o-mini"
    os.environ["OPENAI_API_KEY"]=API_KEYS[f"OpenAI_API_KEYd{api_key_idx}"]
    os.environ["OPENAI_ORG"]=""
    os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepBricks_BASE_URL"]
    ASYNC_MODEL_CONFIG = lm_config.LMConfig(provider="openai_chat", model=model_name)
    for batch_idx in range(BATCH_START,BATCH_END+1):
        src_file=f"res_multi_sample/{subset}/pairs/{subset}_pairs_{batch_idx}.json"
        out_file=f'paraphrase/{subset}/full_{batch_idx}.json'
        os.makedirs(os.path.dirname(out_file),exist_ok=True)
        
        src_data=json.load(open(src_file,"r"))
        if type(used_size) is int:
            src_data=src_data[:used_size]
        ori_id_list,task_list,pos_ans_list,neg_ans_list=load_full_task(data=src_data)
        print(len(task_list))
        
        # logger_file=f"logs_paraphrase/{subset}_{batch_idx}.log"
        # logger=set_logger(logger_file=logger_file)
        logger=None
        asyncio.run(paraphrase(prompt=paraphrase_prompt,model_config=ASYNC_MODEL_CONFIG,
                                    model_name=model_name,
                                num_samples=NUM_SAMPLES,
                                out_file=out_file,logger=logger,
                                task_list=task_list,pos_ans_list=pos_ans_list,
                                neg_ans_list=neg_ans_list,
                                ori_id_list=ori_id_list))

            
        print(f" =============================== time: {time.time()-s_t} =============================== ")
        
    
    
    