import asyncio
import logging
import os
import json
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import random



def add_ori_id(src_file_list,sample_file):
    src_data=[]
    for src_file in src_file_list:
        src_data.extend(json.load(open(src_file,"r")))
    
    curr_data=json.load(open(sample_file,"r"))
    new_data=[]
    for idx,item in tqdm(enumerate(curr_data)):
        new_item={}
        task=item["task"]
        for x in src_data:
            if x["original"]["task"]==task:
                new_item["ori_id"]=x["ori_id"]
            elif "subtask_1" in x and x["subtask_1"]["task"]==task:
                new_item["ori_id"]=x["ori_id"]
                new_item["sub_idx"]=1
            elif "subtask_2" in x and x["subtask_2"]["task"]==task:
                new_item["ori_id"]=x["ori_id"]
                new_item["sub_idx"]=2
            elif "subtask_3" in x and x["subtask_3"]["task"]==task:
                new_item["ori_id"]=x["ori_id"]
                new_item["sub_idx"]=3
            elif "subtask_4" in x and x["subtask_4"]["task"]==task:
                new_item["ori_id"]=x["ori_id"]
                new_item["sub_idx"]=4
        new_item.update(item)
        new_data.append(new_item)
        
    return new_data
    # with open(out_file,"w") as file:
    #     json.dump(new_data,file,indent=4)


def formulate_preference_pair(src_file_list,full_file,sub_file,pairs_file,):
    """
    output format (one entry): 
    {
        "ori_id": "sci_instruct_000000",
        "original": {
            "task": "",
            "pos_ans": "",
            "neg_ans": "",
        },
        "subtask_1": {
            "task": "",
            "pos_ans": "",
            "neg_ans": "",
        },
        "subtask_2": {
            "task": "",
            "pos_ans": "",
            "neg_ans": "",
        }
    }
    """
    
    full_data=add_ori_id(src_file_list,full_file)
    sub_data=add_ori_id(src_file_list,sub_file)
    
    samples_data=full_data+sub_data
    
    print("num of full task:",len(full_data))
    print("num of subtask:",len(sub_data))
    
    if os.path.exists(pairs_file):
        pairs_list=json.load(open(pairs_file,"r"))
        full_sub_pairs={item["ori_id"]:item for item in pairs_list}
    else:
        pairs_list=[]
        full_sub_pairs={}
        
    for item in tqdm(samples_data):
        ori_id=item["ori_id"]
        task=item["task"]

        correct_sample=[item[f"sample{i+1}"] for i in range(NUM_SAMPLE) if item[f"judge{i+1}"]==1]
        incorrect_sample=[item[f"sample{i+1}"] for i in range(NUM_SAMPLE) if item[f"judge{i+1}"]==0]
        if correct_sample!=[]:
            pos_ans=random.choice(correct_sample)
        elif correct_sample==[] and item.get("transfer_ground_truth",None) is not None:
            pos_ans=item["transfer_ground_truth"]
        else:
            continue
        
        if incorrect_sample==[]:
            continue
        else:
            neg_ans=random.choice(incorrect_sample)
            
        if ori_id not in full_sub_pairs.keys():
            full_sub_pairs.update({
                ori_id:{
                    "ori_id":ori_id,
                }
            })
        
        ### subtask
        if "sub_idx" in item:
            sub_sequential=item["sub_idx"]
            full_sub_pairs[ori_id].update({
                f"subtask_{sub_sequential}":{
                    "task":task,
                    "pos_ans":pos_ans,
                    "neg_ans":neg_ans,
                }
            })
            
        ### full task   
        else:
            full_sub_pairs[ori_id].update({
                f"original":{
                    "task":task,
                    "pos_ans":pos_ans,
                    "neg_ans":neg_ans,
                }
            }) 
        
    bad_ori_ids=[]
    er1=0
    er2=0
    for ori_id,item in full_sub_pairs.items():
        if "original" not in item.keys():
            bad_ori_ids.append(ori_id)
            er1+=1
            
        if "original" in item.keys() and len(list(item.keys()))<=2:
            bad_ori_ids.append(ori_id)
            er2+=1
        
        # if "subtask_1" not in item.keys():
        #     bad_ori_ids.append(ori_id)
        # if "subtask_3" in item.keys() and "subtask_2" not in item.keys():
        #     bad_ori_ids.append(ori_id)

    
    
    bad_ori_ids=sorted(list(set(bad_ori_ids)),reverse=True)
    for bad_ori_id in bad_ori_ids:
        full_sub_pairs.pop(bad_ori_id)
        
    print("missing full tasks: ",er1)
    print("0 subtasks: ",er2)
    print("final num: ",len(full_sub_pairs))

    with open(pairs_file,"w") as f:
        json.dump(list(full_sub_pairs.values()),f,indent=4)
        
        
if __name__ =="__main__":
    NUM_SAMPLE=3
    
    dataset="numina_syn_math"
    # batch_idx=0
    ST=22
    END=22
    for batch_idx in range(ST,END+1):
        if dataset!="sci_instruct":
            sampling_file_list=[]
            src_file_list=[f"res/{dataset}/good/{dataset}_{batch_idx}_good.json"]
            full_file=f"res_multi_sample/{dataset}/full_style_transfer/{dataset}_full_{batch_idx}_style_transfer.json"
            sub_file=f"./res_multi_sample/{dataset}/sub_style_transfer/{dataset}_sub_{batch_idx}_style_transfer.json"
            pairs_file=f'./res_multi_sample/{dataset}/pairs/{dataset}_pairs_{batch_idx}.json'
            os.makedirs(os.path.dirname(pairs_file),exist_ok=True)
            
            formulate_preference_pair(src_file_list=src_file_list,full_file=full_file,sub_file=sub_file,pairs_file=pairs_file)
            
            
        else:
            src_file_list=os.listdir("./res/sci_instruct/raw")
            
            full_dir=f'./res_multi_sample/{dataset}/full_checked'
            sub_dir=f'./res_multi_sample/{dataset}/sub_checked'
            
            sampling_file_list=[]
            full_file=f"{full_dir}/{dataset}_full_{batch_idx}_t.json"
            sub_file=f"{full_dir}/{dataset}_sub_{batch_idx}_t.json"
            
            pairs_file=f'./res_multi_sample/{dataset}/pairs_{batch_idx}.json'
            os.makedirs(os.path.dirname(pairs_file),exist_ok=True)
            formulate_preference_pair(src_file_list=src_file_list,full_file=full_file,sub_file=sub_file,pairs_file=pairs_file)
            