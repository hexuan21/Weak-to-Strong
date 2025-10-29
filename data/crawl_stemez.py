import requests
from bs4 import BeautifulSoup
import json
import os
import asyncio
import logging
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from utils_async import *
from zeno_build.models import lm_config
from utils_async import set_logger


def scrape_QA_from_url(url,image_out=True,logger=None):

    response = requests.get(url)
    response.encoding = response.apparent_encoding  

    if response.status_code != 200:
        if logger is not None:
            logger.info(f"url: {url}, failed to retrieve")
        else:
            print(f"url: {url}, failed to retrieve")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    if image_out:
        imgs = soup.find_all('img')
        if len(imgs)>0:
            if logger is not None:
                logger.info(f"url: {url} has image, return None")
            else:
                print(f"url: {url} has image, return None")
            return None   
     
    paragraphs = soup.find_all('p', class_='MsoNormal')
    text_list = []
    
    for p in paragraphs:
        for sub in p.find_all('sub'):
            sub_text = f"_{sub.get_text()}"
            sub.replace_with(sub_text)  

        text_list.append(p.get_text(strip=True))

    return text_list


async def format_norm(text_list,logger=None):
    prompt="The text below contains misplaced line breaks '\\n', which make it appear disjointed and awkward. Your task is to correct the line breaks while keeping the LaTeX syntax intact and ensuring that the content remains unchanged. If necessary, you may place some mathematical formulas on separate lines for better readability. Ensure that the original content is preserved.\n**Output format**:\n### Converted text: <converted text>\n### End\n\nHere’s the input:"
    context_list=[]
    for text in text_list:
        user_input=prompt+text
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = model_config,logger=logger,)

    new_text_list=[]
    for raw_output in raw_outputs:
        sign_1="### Converted text"
        sign_2="### End"
        if sign_1 in raw_output and sign_2 in raw_output:
            new_text=raw_output.split(sign_1)[1].split(sign_2)[0].strip(NOISE_CHARS)
            new_text_list.append(new_text)
        else:
            new_text_list.append(None)

    return new_text_list




async def sci_crawl_one_chap(subject:str,chap_idx,logger=None):
    logger.info(f"Current Subject: =============== {subject} ===============")
    # Contents page of certain chapter in certain subject: https://stemez.com/subjects/science/DPhysics/DPhysics/DPhysics/D00-Ch01.htm
    idx_page_url=f"{sci_url_root}/{subject}/{subject}/{subject}/{subject[0]}00-Ch{chap_idx:02}.htm"
    if subject[0] =="1":
        idx_page_url=f"{sci_url_root}/{subject}/{subject}/{subject}/{subject[:2]}00-Ch{chap_idx:02}.htm"
    if subject[:2]=="1H":
        idx_page_url="https://stemez.com/subjects/science/1HOperationsReseach/1HOperationsReseach/1HOperationsResearch/1H00-Ch01.htm"
    
    text_list=scrape_QA_from_url(idx_page_url,image_out=False)
    que_idx_list=[]
    for row in text_list:
        # PROBLEM09 – 0285:...
        if "PROBLEM" in row and "– " in row and ":" in row:
            QA_item_idx=int(row.split("– ")[1].split(":")[0])
            que_idx_list.append(QA_item_idx)
    if logger is not None:
        logger.info(f"Num of QAs in {subject}-chap{chap_idx:02}: {len(que_idx_list)}")
    else:
        print(f"Num of QAs in {subject}-chap{chap_idx:02}: {len(que_idx_list)}")
        
    raw_que_list=[]
    raw_sol_list=[]
    valid_que_idx_list=[]
    for QA_item_idx in tqdm(que_idx_list):
        # https://stemez.com/subjects/science/DPhysics/DPhysics/DPhysics/D01-0001.htm
        item_url=f"{sci_url_root}/{subject}/{subject}/{subject}/{subject[0]}{chap_idx:02}-{QA_item_idx:04}.htm"
        if subject[0] =="1":
            item_url=f"{sci_url_root}/{subject}/{subject}/{subject}/{subject[:2]}{chap_idx:02}-{QA_item_idx:04}.htm"
        
        raw_rows=scrape_QA_from_url(item_url,image_out=True)
        if raw_rows is None:
            continue
        QA_raw_text="\n".join(raw_rows)
        sign_1=f"PROBLEM{chap_idx:02} – {QA_item_idx:04d}"
        sign_2="Solution"
        if sign_1 in QA_raw_text and sign_2 in QA_raw_text:
            que_raw_text=QA_raw_text.split(sign_1)[1].split(sign_2)[0]
            sol_raw_text=QA_raw_text.split(sign_2)[1]
            raw_que_list.append(que_raw_text)
            raw_sol_list.append(sol_raw_text)
            valid_que_idx_list.append(QA_item_idx)
            
    # que_list= await format_norm(raw_que_list,logger)
    que_list=[raw_que.replace("\n"," ") for raw_que in raw_que_list]
    sol_list= await format_norm(raw_sol_list,logger)
    
    for i in range(len(sol_list)-1,-1,-1):
        if sol_list[i] is None:
            sol_list.pop(i)
            que_list.pop(i)
            valid_que_idx_list.pop(i)
    
    data=[]
    subject_id=subject[1:]
    subject_id=subject_id.lower()
    for idx,(que_idx,que,sol) in enumerate(zip(valid_que_idx_list,que_list,sol_list)):
        data_entry={
            "ori_id":f"stemez_{subject_id}_chap{chap_idx:02d}_{que_idx:04d}",
            "question":que,
            "positive answer":sol,
            "token_len_pos_ans":len(tokenizer.tokenize(sol)),
            "negative answer":"",
        }
        data.append(data_entry)

    res_file=f"./data/STEMEZ/stemez_{subject[1:].lower()}.json"
    os.makedirs(os.path.dirname(res_file),exist_ok=True)
    if not os.path.exists(res_file):
        with open(res_file,"w") as _:
            None
    if os.path.getsize(res_file) > 0:
        previous_data=json.load(open(res_file,'r',encoding='utf-8'))
        previous_data.extend(data)
        logger.info(f"current data size: {len(previous_data)}")
        json.dump(previous_data,open(res_file,'w'),indent=4)
    else:
        json.dump(data,open(res_file,'w'),indent=4)
        logger.info(f"current data size: {len(data)}")
    
    

if __name__ == "__main__":
    logger=set_logger("./logs_other/crawl_stemez.log")
    API_KEYS=json.load(open("./const/api_key.json","r"))
    os.environ["OPENAI_API_KEY"]=API_KEYS["OpenAI_API_KEY5"]
    os.environ["OPENAI_ORG"]=API_KEYS["OpenAI_ORG_ID"]
    
    model_name="gpt-4o-mini"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    sci_url_root="https://stemez.com/subjects/science"
    sci_subject_chapnum_map={
        "DPhysics":37,
        "EChemistry":33,
        "FBiology":31,
        "JMechanics":26,
        "LOrganicChemistry":35,
        "MPhysicalChemistry":15,
        "XGenetics":16,
        "YOptics":27,
        
        "GComputerScience":25,
        "ZEconomics":33,
        "1BBusinessMgmt":32,
        "1HOperationsReseach":13,
        "1JPsychology":21,
    }
    
    
    subject="1BBusinessMgmt"
    for chap_idx in range(1,sci_subject_chapnum_map[subject]+1):
        asyncio.run(sci_crawl_one_chap(subject,chap_idx,logger))
    







