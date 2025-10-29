"""From https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py."""
"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
import json
from typing import Any
from datetime import datetime
import aiolimiter
import openai
from openai import AsyncOpenAI,OpenAI

from tqdm.asyncio import tqdm_asyncio

from zeno_build.models import lm_config

        

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    num_samples:int=None,
    logger=None,
) -> dict[str, Any]:
    if os.environ.get("OPENAI_ORG") is not None:
        client = AsyncOpenAI(organization=os.environ["OPENAI_ORG"],api_key=os.environ["OPENAI_API_KEY"])
    else:
        client = AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"])
    
    async with limiter:
        for _ in range(3):
            try:
                if num_samples is not None:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=num_samples,
                        top_p=top_p,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                if logger is not None:
                    logger.info(f"\n{response.usage}")
                else:
                    print(f"\n{response.usage}")
                await client.close()
                return response.to_dict()
               
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(5)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(5)
            except openai.BadRequestError:
                logging.warning("OpenAI API Invalid Request: Prompt was filtered")
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except openai.APIConnectionError:
                logging.warning(
                    "OpenAI API Connection Error: Error Communicating with OpenAI"
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    full_contexts: list[dict],
    model_config: lm_config.LMConfig,
    temperature: float = 0.7,
    max_tokens: int = 8000,
    top_p: float = 1,
    requests_per_minute: int = 80,
    tqdm: bool = True,
    num_samples: int = 1,
    logger=None,
    
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_ORG") is None or os.environ.get("OPENAI_ORG")=="":
        openai.base_url=os.environ["OPENAI_BASE_URL"]
    else:
        openai.organization = os.environ["OPENAI_ORG"]
    
    
    if logger is not None:
        logger.info(f"model_name: {model_config.model}")
        logger.info(f"temperature: {temperature}")
        logger.info(f"max_tokens: {max_tokens}")
        logger.info(f"top_p: {top_p}")
        logger.info(f"requests_per_minute: {requests_per_minute}")
    else:
        print(f"model_name: {model_config.model}")
        print(f"temperature: {temperature}")
        print(f"max_tokens: {max_tokens}")
        print(f"top_p: {top_p}")
        print(f"requests_per_minute: {requests_per_minute}")
    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config.model,
            messages=full_context["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            logger=logger,
            num_samples=num_samples
        )
        for full_context in full_contexts
    ]
    if tqdm:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    
    if num_samples!=1:
        samples_2dlist=[]
        for response in responses:
            samples=[response["choices"][i]["message"]["content"] for i in range(len(response["choices"]))]
            samples_2dlist.append(samples)
        return samples_2dlist
    else:
        
        res=[response["choices"][0]["message"]["content"] for response in responses]
        for idx, x in enumerate(res):
            if (len(x)) < 1000:
                print(f"xxxxxxxxxxxx {x}")
                print(f"bad index {res.index(x)}")
        return res



NOISE_CHARS="#.*: \n"

### -----------------------------------------------------------------------------------
### preprocess before running pipeline
### -----------------------------------------------------------------------------------
def init_client(model_name,key_idx):
    client = None
    API_KEYS=json.load(open("./const/api_key.json","r"))
    if "gpt" in model_name:
        API_KEY=API_KEYS[f"OpenAI_API_KEY{key_idx}"]
        ORG_ID=API_KEYS["OpenAI_ORG_ID"]
        client = OpenAI(organization=ORG_ID,api_key=API_KEY)
    if model_name=="deepseek-coder" or model_name == "deepseek-chat":
        API_KEY=API_KEYS["DeepSeek_API_KEY"]
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client


def set_logger(logger_file="./logs/test.log"):
    now = datetime.now()
    date_time = now.strftime("%m-%d--%H-%M-%S")
    logger_file=logger_file.replace(".log",f"_{date_time}.log")
    os.makedirs(os.path.dirname(logger_file),exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler(logger_file,encoding='utf-8'), 
                        logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    class HttpxFilter(logging.Filter):
        def filter(self, record):
            return "httpx" not in record.name
    logger.addFilter(HttpxFilter())
    return logger


def mapping_token_len_num_subtask(logger,len):
    
    if len<=400:
        # logger.info("full_pos_ans is too short")
        return None
    if len>400 and len<=1000:
        return 2
    if len>1000 and len <=1600:
        return 3
    if len>1600:
        # logger.info("full_pos_ans is too long")
        return None


def s0_filter_original_task(logger,full_task, full_pos_ans):
    TRIGGER_WORDS=["figure","skematic","diagram","image","plot","picture","http","url","[asy]","[/asy]","tickleng"]
    # the following words are added to filter out the problems with multiple sub-problems in cn_k12
    for word in TRIGGER_WORDS+["Ⅱ","II","(II)","2. ","(2) ",]:
        if word in full_task.lower():
            logger.info(f"Error: {word} exists in task description, this task will be skipped.")
            return 0
        
    for word in TRIGGER_WORDS:
        if word in full_pos_ans.lower():
            logger.info(f"Error: {word} exists in solution, this task will be skipped.")
            return 0
    
    # SEPE_WORDS=["(2)","Ⅱ."]
    # for word in SEPE_WORDS:
    #     if word in full_task.lower():
    #         logger.info(f"{word} exists in task description, this task will be skipped.")
    #         return 0
    
    OTHER_LANG_LETTERS=['é', 'è', 'ê', 'ë', 'à', 'ù', 'î', 'ï', 'ô', 'û', 'ç'] #French
    OTHER_LANG_LETTERS+=['ä', 'ö', 'ü', 'ß'] #German
    OTHER_LANG_LETTERS+=['á', 'à', 'â', 'ã', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ç'] #Spanish
    OTHER_LANG_LETTERS=list(set(OTHER_LANG_LETTERS))
    
    if any(char in full_task for char in OTHER_LANG_LETTERS) \
        or any(char in full_pos_ans for char in OTHER_LANG_LETTERS):
            logger.info(f"Error: the task or solution is not English, this task will be skipped.")
            return 0
    
    # from langdetect import detect
    
    # lang=detect(full_task)
    # if lang!="en" and lang!="ca":
    #     logger.info(f"Error: the task is in {lang}, not English, this task will be skipped.")
    #     return lang
    
    # lang=detect(full_pos_ans)
    # if lang != "en":
    #     logger.info(f"Error: the solution is in {lang}, not English, this task will be skipped.")
    #     return 0
    
    return 1


async def convert_MCQ_to_open(logger,model_config,task_list,sol_list):
    prompt="Here we provide a Multiple Choice Question (MCQ) and its solution, please try to convert it into an open-ended question, removing the options in question text and making corresponding modification in solution text. If you think it's hard and impossible to convert, just output N/A. \nOutput format: \n### Question: <converted question>\n### Solution: <converted solution>\n### End.\n\nHere's the input:"
    context_list=[]
    for task,solution in zip(task_list,sol_list):
        user_input=prompt+"\n### MCQ-format Question: \n"+task+"\n### Original solution: \n"+solution
        context=dict(messages=[{
                "content":user_input,
                "role":"user"}] )
        context_list.append(context)
    raw_outputs = await generate_from_openai_chat_completion(
        full_contexts = context_list,model_config = model_config, logger=logger,)

    new_task_list=[]
    new_sol_list=[]
    for raw_output in raw_outputs:
        sign_1="### Question"
        sign_2="### Solution"
        sign_3="### End"
        if sign_1 in raw_output and sign_2 in raw_output:
            new_task=raw_output.split(sign_1)[1].split(sign_2)[0].strip(NOISE_CHARS)
            new_task_list.append(new_task)
            new_sol=raw_output.split(sign_2)[1].strip(NOISE_CHARS)
            new_sol_list.append(new_sol)
        else:
            new_task_list.append(None)
            new_sol_list.append(None)
    for idx,_ in enumerate(new_sol_list):
        if sign_3 in new_sol_list[idx]:
            new_sol_list[idx].replace(sign_3,"")
            new_sol_list[idx].strip(NOISE_CHARS)
    
    return new_task_list,new_sol_list



### -----------------------------------------------------------------------------------
### match output in each stage
### -----------------------------------------------------------------------------------
def s1_res_match(logger,s1_res):
    subtask_list=[]
    sub_pos_ans_list=[]

    sign_end="### Sign of End"
    num_subtask=s1_res.count("### Item")
    logger.info(f"\nnum of subtasks: {num_subtask}")
    for idx in range(num_subtask):
        former_sign=f"### Item {idx+1}"
        latter_sign=f"### Item {idx+2}"
        
        if idx==num_subtask-1:
            matched=former_sign in s1_res
        else:
            matched=former_sign in s1_res and latter_sign in s1_res
        
        sign_task=f"New problem {idx+1}"
        sign_solution=f"New solution {idx+1}"
        
        sign_task_v2=f"New Problem {idx+1}"
        sign_solution_v2=f"New Solution {idx+1}"
        
        if matched:
            if idx==num_subtask-1:
                chunk=s1_res.split(former_sign)[1]
                if sign_end in chunk:
                    chunk.replace(sign_end,"")
            else:
                chunk=s1_res.split(former_sign)[1].split(latter_sign)[0]
                
            chunk=chunk.strip(NOISE_CHARS)
            if sign_task in chunk and sign_solution in chunk:
                subtask=chunk.split(sign_task)[1].split(sign_solution)[0]
                sub_pos_ans=chunk.split(sign_solution)[1]
                subtask_list.append(subtask)
                sub_pos_ans_list.append(sub_pos_ans)
            elif sign_task_v2 in chunk and sign_solution_v2 in chunk:
                subtask=chunk.split(sign_task_v2)[1].split(sign_solution_v2)[0]
                sub_pos_ans=chunk.split(sign_solution_v2)[1]
                subtask_list.append(subtask)
                sub_pos_ans_list.append(sub_pos_ans)
            else:
                logger.info("\nstage1, no matched sign_task and sign_solution")
                return None, None
            
        else:
            logger.info("\nstage1, no matched chunk_sign")
            return None, None
    
    for idx, (task, pos_ans) in enumerate(zip(subtask_list,sub_pos_ans_list)):
        subtask_list[idx]=task.strip(NOISE_CHARS)
        sub_pos_ans_list[idx]=pos_ans.strip(NOISE_CHARS)
        
    return subtask_list,sub_pos_ans_list



def s2_res_match(logger,s2_res):
    sign_1="### Incorrect solution"
    sign_end="### End"
                
    if sign_1 in s2_res:
        sub_neg_ans=s2_res.split(sign_1)[1]
        if sign_end in sub_neg_ans:
            sub_neg_ans.replace(sign_end,"")
        return sub_neg_ans.strip(NOISE_CHARS)
    else:
        logger.info("\nstage2, no matched chunk_sign")
        return None



def s3_res_match(logger,s3_res):
    sign_1="### Full negative answer for original problem"
    sign_end="### End"
    
    if sign_1 in s3_res:
        if sign_end in s3_res:
            full_neg_ans=s3_res.split(sign_1)[1].split(sign_end)[0]
        else:
            full_neg_ans=s3_res.split(sign_1)[1]
        return full_neg_ans.strip(NOISE_CHARS)
    else:
        logger.info("\nstage3, no matched chunk_sign for full-neg-ans")
        return None
    
    
    
### -----------------------------------------------------------------------------------
### save to json
### -----------------------------------------------------------------------------------

def save_res(res_file,ori_id,original_task,full_pos_ans,full_neg_ans,subtask_list,sub_pos_ans_list,sub_neg_ans_list,error_type_list=[],full_pos_len=None,):
    data_entry={
        "ori_id":ori_id,
        "original":
            {
                "task":original_task,
                "pos_ans":full_pos_ans,
                "neg_ans":full_neg_ans,
                "token_len_pos_ans":full_pos_len,
            },
    }
    
    if not len(subtask_list)==len(sub_pos_ans_list) and len(subtask_list)==len(sub_neg_ans_list):
        print("missing or duplicate items in list of sub-tasks of sub-answers")
        return

    if sub_pos_ans_list==[]:
        sub_pos_ans_list=["" for _ in range(len(subtask_list))]
    if sub_neg_ans_list==[]:
        sub_neg_ans_list=["" for _ in range(len(subtask_list))]
    
    for sub_idx,_ in enumerate(subtask_list):
        if error_type_list!=[]:
            if error_type_list[sub_idx] in [1,"1"]:
                error_type="computation error"  
            else:
                error_type="method error"
        else:
            error_type=None
        
        data_entry.update({
            f"subtask_{sub_idx+1}":
                {
                    "task":subtask_list[sub_idx],
                    "pos_ans":sub_pos_ans_list[sub_idx],
                    "neg_ans":sub_neg_ans_list[sub_idx],
                    "error_type":error_type,
                }
        })

    if not os.path.exists(res_file):
        with open(res_file,"w") as _:
            None
    
    if os.path.getsize(res_file) > 0:
        data=json.load(open(res_file,'r',encoding='utf-8'))
        data.append(data_entry)
        json.dump(data,open(res_file,'w'),indent=4)
    else:
        json.dump([data_entry],open(res_file,'a'),indent=4)
        
        
def save_res_list(res_file,ori_id_list,full_task_list,full_pos_ans_list,full_neg_ans_list,full_pos_len_list,
                  subtask_2dlist,sub_pos_ans_2dlist,sub_neg_ans_2dlist,error_type_2dlist=[],): 
    if full_neg_ans_list==[]:
        full_neg_ans_list=["" for _ in range(len(ori_id_list))]
        
    if full_pos_len_list==[]:
        full_pos_len_list=["" for _ in range(len(ori_id_list))]
    
    if sub_neg_ans_2dlist==[]:
        sub_neg_ans_2dlist=[[] for _ in range(len(ori_id_list))]
    if error_type_2dlist==[]:
        error_type_2dlist=[[] for _ in range(len(ori_id_list))]
    
    # Assert: len(ori_id_list)==len(full_task_list)==...==len(sub_neg_ans_2dlist)
    
    all_data=[]
    
    for idx in range(len(ori_id_list)):
        ori_id=ori_id_list[idx]
        full_task=full_task_list[idx]
        full_pos_ans=full_pos_ans_list[idx]
        full_neg_ans=full_neg_ans_list[idx]
        full_pos_len=full_pos_len_list[idx]
        subtask_list=subtask_2dlist[idx]
        sub_pos_ans_list=sub_pos_ans_2dlist[idx]
        sub_neg_ans_list=sub_neg_ans_2dlist[idx]
        error_type_list=error_type_2dlist[idx]
        
        data_entry={
            "ori_id":ori_id,
            "original":
                {
                    "task":full_task,
                    "pos_ans":full_pos_ans,
                    "neg_ans":full_neg_ans,
                    "token_len_pos_ans":full_pos_len,
                },
        }
        
        if sub_neg_ans_list==[]:
            sub_neg_ans_list=["" for _ in range(len(subtask_list))]
        if error_type_list==[]:
            error_type_list=["" for _ in range(len(subtask_list))]
        
        for sub_idx in range(len(subtask_list)):
            if error_type_list!=[]:
                if error_type_list[sub_idx] in [1,"1"]:
                    error_type="computation error"  
                else:
                    error_type="method error"
            else:
                error_type=None
            data_entry.update({
                f"subtask_{sub_idx+1}":
                    {
                    "task":subtask_list[sub_idx],
                    "pos_ans":sub_pos_ans_list[sub_idx],
                    "neg_ans":sub_neg_ans_list[sub_idx],
                    "error_type":error_type
                }
            })
        
        all_data.append(data_entry)
        
    if not os.path.exists(res_file):
        with open(res_file,"w") as _:
            None
    
    if os.path.getsize(res_file) > 0:
        data=json.load(open(res_file,'r',encoding='utf-8'))
        data.extend(all_data)
        json.dump(data,open(res_file,'w'),indent=4)
    else:
        json.dump(all_data,open(res_file,'w'),indent=4)