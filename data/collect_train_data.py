import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datasets import load_dataset

from transformers import AutoTokenizer



def collect_prm_long():
    
    dataset="PRM"
    data=[json.loads(line) for line in open("./data/PRM/raw_data/prm800k.jsonl","r")]

    hard_data=[]
    pos_token_num_list=[]

    for idx,item in tqdm(enumerate(data)):
        que=item["chosen"][0]["content"]
        pos_ans=item["chosen"][1]["content"]
        neg_ans=item["rejected"][1]["content"]
        pos_tokens=tokenizer.tokenize(pos_ans)
        
        pos_token_num_list.append(len(pos_tokens))
        if len(pos_tokens)>=LEN_MIN and len(pos_tokens)<=LEN_MAX:
            hard_data.append({
                "ori_id":f"prm_{idx:06d}",
                "question":que,
                "positive answer":pos_ans,
                "token_len_pos_ans":len(pos_tokens),
                "negative answer":neg_ans,
            })

    print(f"Num of hard problems in {dataset}",len(hard_data))
    
    save_path=f"./data/PRM/prm_long.json"
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path,"w") as file:
        json.dump(hard_data,file,indent=4)


    plt.figure(figsize=(8, 6))  
    plt.hist(pos_token_num_list, bins=50, edgecolor='black')  
    plt.xlabel('Number of tokens of positive answer')  
    plt.ylabel('Frequency')  
    plt.title('Histogram of Random Data')  
    plt.grid(True)  
    plt.tight_layout()  
    plt.savefig(f"./data/{dataset}/token_num_dist.png")



def collect_math(split="train"):
    dataset="MATH"
    
    src_dir=f"./data/MATH/MATH_original/{split}"

    hard_data=[]
    pos_token_num_list=[]
    
    for sub_dir in os.listdir(src_dir):
        sub_hard_data=[]
        for json_file in tqdm(sorted(os.listdir(os.path.join(src_dir,sub_dir)))):
            ori_idx=json_file.split(".")[0]
            data=json.load(open(os.path.join(src_dir,sub_dir,json_file),"r",encoding='utf-8'))
            
                         
            if "4" in data["level"] or "5" in data["level"]:
                pos_ans=data["solution"]
                pos_tokens=tokenizer.tokenize(pos_ans)
                pos_token_num_list.append(len(pos_tokens))
                if len(pos_tokens)>=LEN_MIN and len(pos_tokens)<=LEN_MAX:
                    sub_hard_data.append({
                        "ori_idx":ori_idx,
                        "source":f"MATH_{sub_dir}",
                        "question":data["problem"],
                        "positive answer":data["solution"],
                        "negative answer":"",
                        "token_len_pos_ans":len(pos_tokens),
                    })
        print(f"{sub_dir}, {len(sub_hard_data)}")
        hard_data.extend(sub_hard_data)

    print(f"Num of hard problems in {dataset}",len(hard_data))

    save_path=f"./data/{dataset}/MATH_train_hard.json"
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path,"w") as file:
        json.dump(hard_data,file,indent=4)
    
    plt.figure(figsize=(8, 6))  
    plt.hist(pos_token_num_list, bins=50, edgecolor='black')  
    plt.xlabel('Number of tokens of positive answer')  
    plt.ylabel('Frequency')  
    plt.title('Histogram of Random Data')  
    plt.grid(True)  
    plt.tight_layout()  
    plt.savefig(f"./data/{dataset}/token_num_dist.png")




def collect_numina_math():
    dataset="NuminaMath"
    source_list=["cn_k12","synthetic_math","orca_math","olympiads","synthetic_amc","aops_forum"]
    data = load_dataset("AI-MO/NuminaMath-CoT",split="train")
    
    for source in source_list:    
        sub_data=[]
        for item in tqdm(data):
            if item["source"] in [source]:
                sub_data.append(item)
                
        hard_data=[]
        pos_token_num_list=[]
        
        for idx,item in tqdm(enumerate(sub_data)):
            pos_ans=item["solution"]
            pos_tokens=tokenizer.tokenize(pos_ans)
            pos_token_num_list.append(len(pos_tokens))
            if len(pos_tokens)>=LEN_MIN and len(pos_tokens)<=LEN_MAX:
                hard_data.append({
                    "ori_id":f"{item['source']}_{idx:06d}",
                    "question":item["problem"],
                    "positive answer":pos_ans,
                    "token_len_pos_ans":len(pos_tokens),
                    "negative answer":"",
                })

        print(f"Num of hard problems in subset {source} of {dataset}",len(hard_data))

        save_path=f"./data/{dataset}/{source}_train_hard.json"
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path,"w") as file:
            json.dump(hard_data,file,indent=4)
            
        
        plt.figure(figsize=(8, 6))  
        plt.hist(pos_token_num_list, bins=50, edgecolor='black')  
        plt.xlabel('Number of tokens of positive answer')  
        plt.ylabel('Frequency')  
        plt.title('Histogram of Random Data')  
        plt.grid(True)  
        plt.tight_layout()  
        plt.savefig(f"./data/{dataset}/{source}_token_num_dist.png")


def collect_sci_inst():
    dataset="SciInstruct"
    all_dataset = load_dataset("zd21/SciInstruct")

    def filter_by_attribute(example, attribute, values):
        return example[attribute] in values
    raw_data = all_dataset['train'].filter(lambda x: filter_by_attribute(x, "subject", ["physics_chemistry",]))
    raw_data=[item for item in raw_data]
    print(len(raw_data))

    token_len_list=[]    
    hard_data=[]

    
    for idx,item in tqdm(enumerate(raw_data)):
        
        pos_token_len=len(tokenizer.tokenize(item['summary']))
        token_len_list.append(pos_token_len)
        
        if pos_token_len>=LEN_MIN and pos_token_len<=LEN_MAX:
            hard_data.append({
                "ori_id":f"{dataset}_{idx:06d}",
                "question":item["content"],
                "positive answer":item["summary"],
                "token_len_pos_ans":pos_token_len,
                "negative answer":"",
            })
    
    save_path=f"./data/{dataset}/phy_chem.json"
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path,"w") as file:
        json.dump(hard_data,file,indent=4)
    
    plt.figure(figsize=(8, 6))  
    plt.hist(token_len_list, bins=50, edgecolor='black')  
    plt.xlabel('Number of tokens of positive answer')  
    plt.ylabel('Frequency')  
    plt.title('Histogram of Random Data')  
    plt.grid(True)  
    plt.tight_layout()  
    plt.savefig(f"./data/{dataset}/token_num_dist.png")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    LEN_MIN=0
    LEN_MAX=1e10
    # collect_numina_math()
    
    # LEN_MIN=500
    # LEN_MAX=2000
    # collect_math()
    
    # LEN_MIN=600
    # LEN_MAX=2000
    # collect_prm_long()
    
    # LEN_MIN=500
    # LEN_MAX=2000
    collect_sci_inst()
