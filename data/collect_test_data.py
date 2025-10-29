import asyncio
import logging
import os
import json
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import random

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from zeno_build.models import lm_config

from utils_async import generate_from_openai_chat_completion,set_logger,NOISE_CHARS


def MATH_test500():
    all_qa=[]
    f="data/MATH/MATH_test500.jsonl"
    data=[json.loads(line) for line in open(f,"r")]
    for x in data:
        all_qa.append({
           "question":x["problem"],
           "ground_truth":x["solution"]
        }) 
    with open(f"{test_dir}/MATH_test500.json","w") as f:
        json.dump(all_qa,f,indent=4)
    
    


def numina_aops_forum():
    path="./data/NuminaMath/aops_forum.json"
    data=json.load(open(path,"r"))
    new_data=[]
    for item in tqdm(data):
        if item["token_len_pos_ans"]<SOLUTION_LEN_MIN:
            continue
        que=item["question"]
        sol=item["positive answer"]
        new_data.append({'question':que,
                         "ground_truth":sol})
    
    with open(f"{test_dir}/aops_forum.json","w") as f:
        json.dump(new_data,f,indent=4)


def numina_olympiads():
    path="./data/NuminaMath/olympiads.json"
    data=json.load(open(path,"r"))
    new_data=[]
    for item in tqdm(data):
        if item["token_len_pos_ans"]<SOLUTION_LEN_MIN:
            continue
        que=item["question"]
        sol=item["positive answer"]
        new_data.append({'question':que,
                         "ground_truth":sol})
    
    with open(f"{test_dir}/olympiads.json","w") as f:
        json.dump(new_data,f,indent=4)
    

def jee_bench():
    path="data/JEE_Bench_dataset.json"
    data=json.load(open(path,"r"))
    new_data=[{"question":x["question"],"ground_truth":x["gold"]} for x in data if x["subject"]=="math"]
    print(len(new_data))
    with open(f"{test_dir}/jee_bench.json","w") as f:
        json.dump(new_data,f,indent=4)


def olym_arena():
    from datasets import load_dataset
    dataset = load_dataset("GAIR/OlympicArena", "Math", split="val")

    items=[]
    for x in dataset:
        if x["figure_urls"] != None:
            continue
        items.append({
            "question":x["problem"],
            "ground_truth":x["solution"]
        })
    
    print(len(items))
    with open(f"{test_dir}/olympic_arena.json","w") as f:
        json.dump(items,f,indent=4)
    

def mmlu_pro_math():
    mapping={
        "A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,
    }
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items=[]
    for x in dataset:
        if x["category"] != "math":
            continue
        bad_word_sign=0
        for w in ["statement","following","which"]:
            if w in x["question"].lower():
                bad_word_sign=1
                break
        if bad_word_sign:
            continue
        
        correct_idx=mapping[x["answer"]]
        items.append({
            "question":x["question"],
            "ground_truth":x["options"][correct_idx]
        })
    
    print(len(items))
    with open(f"{test_dir}/mmlu_pro_math.json","w") as f:
        json.dump(items,f,indent=4)



def agi_eval(set_name):
    path=f"data/AGI_Eval/{set_name}.jsonl"
    data=[json.loads(line) for line in open(path,"r",encoding="utf-8")]
    items=[]
    for x in data:
        if set_name=="sat-math":
            que=x["question"]+"\nOptions: \n"+f"{x['options']}"
            sol=x["other"]["solution"]
        elif set_name=="gaokao-mathcloze":
            que=x["question"]
            sol=x["answer"]
        elif set_name=="gaokao-mathqa":
            que=x["question"]+"\nOptions: \n"+f"{x['options']}"
            sol=x["label"]
        elif set_name=="aqua-rat":
            que=x["question"]+"\nOptions: \n"+f"{x['options']}"
            sol=x["other"]["solution"]
        elif set_name=="amc-aime":
            que=x["question"]
            sol=x["other"]["solution"]
        else:
            print("set_name not supported, exit")
            exit()
        
        items.append({
            "question":que,
            "ground_truth":sol
        })
        
    print(len(items))
    with open(f"{test_dir}/agi_eval_{set_name}.json","w") as f:
        json.dump(items,f,indent=4)



async def translate_cn_en(src_file,out_file):
    model_name="gpt-4o-mini"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)

    API_KEYS=json.load(open("./const/api_key.json","r"))
    os.environ["OPENAI_API_KEY"]=API_KEYS[f"OpenAI_API_KEYd1"]
    os.environ["OPENAI_ORG"]=""
    os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepBricks_BASE_URL"]
    
    sign_1="New Question"
    sign_2="New Solution/Answer"
    
    prompt=f"Here we provide a math Muliple-Choice Question and its solution in Chinese, please translate it into English, DO NOT change the content and meaning of the problem ans solution. \nOutput format: \n### {sign_1}: <translated question>\n### {sign_2}: <translated solution/answer>. Here's the input: "
    
    data=json.load(open(src_file,"r"))
    que_list=[]
    sol_list=[]
    for x in data:
        que_list.append(x["question"])
        sol_list.append(x["ground_truth"])
    print(len(que_list))
    
    context_list=[]
    for que,sol in zip(que_list,sol_list):
        user_input=prompt+"\n### Question: \n"+que+"\n### Solution/Answer: \n"+sol
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
        
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = model_config)
    
    new_data=[]
    for idx, raw_output in enumerate(raw_outputs):
        if sign_1 not in raw_output or sign_2 not in raw_output:
            continue
        
        new_que=raw_output.split(sign_1)[1].split(sign_2)[0].strip(NOISE_CHARS)
        new_sol=raw_output.split(sign_2)[1].strip(NOISE_CHARS)
        new_data.append({"question":new_que,
                         "ground_truth":new_sol})
        
    print(len(new_data))
    with open(out_file,"w") as f:
        json.dump(new_data,f,indent=4)



async def convert_MCQ(src_file, out_file):
    model_name="gpt-4o-mini"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)

    API_KEYS=json.load(open("./const/api_key.json","r"))
    os.environ["OPENAI_API_KEY"]=API_KEYS[f"OpenAI_API_KEYd1"]
    os.environ["OPENAI_ORG"]=""
    os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepBricks_BASE_URL"]
    
    data=json.load(open(src_file,"r"))
    que_list=[]
    sol_list=[]
    for x in data:
        if "Then\n\n(A)" in x["question"] or "Then\n\nA" in x["question"]:
            continue
        que_list.append(x["question"])
        sol_list.append(x["ground_truth"])
    print(len(que_list))
    
    sign_1="New Question"
    sign_2="New Solution/Answer"
    
    prompt=f"Here we provide a math problem and its answer, if it's an Multiple-Choice Question, please first remove the options and then convert it to an **Open-Ended** question with '?'. Otherwise (like fill-in-the-blank questions), only repeat the question and answer in your output. \nFor those MCQs, please note that if the options are statements like 'area of the triangle $X Y Z$ is $6 \\sqrt6$' or '$\\quad g^\prime(2)=\\frac(1)(15)$' just output 'None'; If the original solution/answer is only an option sequential like 'A', 'C', or multiple-choice like 'BC', 'AC', please only replace it with the content of option in your output.  \nOutput format: \n### {sign_1}: <converted question>\n### {sign_2}: <content of the correct option>. Here's the input: "
    
    context_list=[]
    for que,sol in zip(que_list,sol_list):
        user_input=prompt+"\n### Question: \n"+que+"\n### Solution/Answer: \n"+sol
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
        
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = model_config)
    
    new_data=[]
    bad_data=[]
    for idx, raw_output in enumerate(raw_outputs):
        if sign_1 not in raw_output or sign_2 not in raw_output:
            continue
        new_que=raw_output.split(sign_1)[1].split(sign_2)[0].strip(NOISE_CHARS)
        new_sol=raw_output.split(sign_2)[1].strip(NOISE_CHARS)
        
        bad_word_sign=0
        for w in ["none","statement","following"]:
            if w in raw_output.lower():
                bad_word_sign=1
                break
        if bad_word_sign:
            bad_data.append({"question":new_que,
                         "ground_truth":new_sol})
            continue
        
        new_data.append({"question":new_que,
                         "ground_truth":new_sol})
        
    print(len(new_data))
    with open(out_file,"w") as f:
        json.dump(new_data,f,indent=4)

    with open(out_file.replace(".json","_bad.json"),"w") as f:
        json.dump(bad_data,f,indent=4)

if __name__ == "__main__":
    test_dir="test_data"
    os.makedirs(test_dir,exist_ok=True)
    SOLUTION_LEN_MIN=0
    # numina_aops_forum()
    # numina_olympiads()
    # MATH_test500()
    # jee_bench()
    # olym_arena()
    # mmlu_pro_math()
    # agi_eval("sat_math")
    # agi_eval("gaokao_mathcloze")
    # agi_eval("gaokao_mathqa")
    # agi_eval("amc-aime")
    # agi_eval("aqua-rat")

    ################################### Translate CN to EN ##################################
    # set_name="agi_eval_gaokao-mathqa"
    # set_name="agi_eval_gaokao-mathcloze"
    # src_file=f"test_data/{set_name}.json"
    # out_file=f"test_data/{set_name}.json"
    # asyncio.run(translate_cn_en(src_file=src_file,out_file=out_file))    
    
    
    ################################### Convert MCQ to QA ###################################
    # set_name="agi_eval_gaokao-mathqa"
    # set_name="agi_eval_sat-math"
    # set_name="jee_bench"
    # src_file=f"test_data/{set_name}.json"
    # out_file=f"test_data/converted_{set_name}.json"
    # asyncio.run(convert_MCQ(src_file=src_file,out_file=out_file))
    
    