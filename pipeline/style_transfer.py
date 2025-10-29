import json
import os
import argparse
import random
import tiktoken
from tqdm import tqdm
import concurrent.futures
from openai import AsyncOpenAI,OpenAI



def fetch_prediction_cohere(d):
    return [
        d,
        client.chat(
            model="command-r-08-2024",
            chat_history=[
                {"role": "SYSTEM", "message": "You're a model that helps generate a solution with the help of reference answer.\n\nYou can refer to the provided information, pretend you haven't seen it, and generate a concise solution from scratch, with a little bit rephrasing."},
            ],
            message=d["task"] + "\n\n" + "Reference Info:\n" + d["ground_truth"] + "\n\n" + "Let's think step by step from scratch! (Use the symbols you commonly utilize in the solution, such as Latex symbols. Don't follow the symbols in the reference info. Use less **Conclusion:** paragraph title in the end. Use less 'Let's' or '##' to start with in the first solution sentence.)",
            temperature=1,
            max_tokens=1000
        ).text
    ]


def fetch_prediction_db(d):
    sample_token=[]
    for k,v in d.items():
        if "sample" in k:
            sample_token.append(len(encoding.encode(v)))            
    sample_token_num=int(sum(sample_token)/len(sample_token))
    
    return [
        d,
        client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", 
                     "content": "You're a model that helps generate a solution with the help of reference answer.\n\nYou can refer to the provided information, pretend you haven't seen it, and generate a concise solution from scratch, with a little bit rephrasing."},
                    {"role": "user", 
                     "content": "Problem: \n"+d["task"] + "\n\n" + "Reference Info:\n" + d["ground_truth"] + "\n\n" + f"Let's think step by step from scratch! (Use the symbols you commonly utilize in the solution, such as Latex symbols. Don't follow the symbols in the reference info. Use less **Conclusion:** paragraph title in the end. Use less 'Let's' or '##' to start with in the first solution sentence. NOTE that the number of tokens of your solution SHOULD be in range of {int(sample_token_num*0.8)} to {int(sample_token_num*1.2)})"},
                ],
                stream=False,
                temperature=1.0,
                max_tokens=1000
                
            ).choices[0].message.content
    ]

def style_transfer(src_file,out_file):
    src_data = json.load(open(src_file,"r",encoding='utf-8'))
    style_transfer_data, style_transfer_tasks = list(), list()
    for d in src_data:
        if not d["judge1"] and not d["judge2"] and not d["judge3"]:
            style_transfer_data.append(d)
            style_transfer_tasks.append(d["task"])
    
    predictions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        tasks = []
        for d in tqdm(style_transfer_data):
            tasks.append(executor.submit(fetch_prediction_db, d))
        
        for future in tqdm(concurrent.futures.as_completed(tasks)):
            predictions.append(future.result())
    
    outputs = list()
    for pred in predictions:
        pred[0]["transfer_ground_truth"] = pred[1]
        outputs.append(pred[0])
    
    for d in src_data:
        if d["task"] not in style_transfer_tasks:
            d["transfer_ground_truth"]=None
            outputs.append(d)
    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=4)



if __name__ == "__main__":
    API_KEYS=json.load(open("./const/api_key.json","r"))
    model_name="gpt-4o-mini"
    
    os.environ["OPENAI_API_KEY"]=API_KEYS["OpenAI_API_KEYd1"]
    os.environ["OPENAI_ORG"]=""
    os.environ["OPENAI_BASE_URL"]=API_KEYS["OpenAI_BASE_URL"]
    client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

    encoding = tiktoken.get_encoding("cl100k_base")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_or_subtask",default="sub",type=str,)
    parser.add_argument("--set_name",default="numina_cn_k12",type=str,)
    parser.add_argument("--batch_idx",default=0,type=int,)
    args = parser.parse_args()
    set_name=args.set_name
    full_or_subtask=args.full_or_subtask
    batch_idx=args.batch_idx
    
    ST=62
    END=63
    for batch_idx in range(ST,END+1):
        root_dir=f"res_multi_sample/{set_name}"
        if full_or_subtask in ["full","sub"]:
            src_file = f"{root_dir}/{full_or_subtask}_checked/{set_name}_{full_or_subtask}_{batch_idx}.json"
        else:
            raise ValueError("full_or_subtask")

        out_file=f"{root_dir}/{full_or_subtask}_style_transfer/{set_name}_{full_or_subtask}_{batch_idx}_style_transfer.json"
        os.makedirs(os.path.dirname(out_file),exist_ok=True)
        
        style_transfer(src_file=src_file,out_file=out_file)