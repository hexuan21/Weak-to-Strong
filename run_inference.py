import re
import ast
import json
import csv
import pandas as pd
import argparse
import logging
import os

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

torch.manual_seed(42)

def clean_generated_text(generated_text):
    stop = ['system', 'user', 'assistant', 'Alice:']
    clean_string = re.split('|'.join(stop), generated_text)[0].strip()
    return clean_string


def apply_chat_template(tokenizer, data):
    chat_templated_texts, gold_answers = list(), list()
    for orig_msg in data:
        # print("orig_msg",orig_msg)
        if "meta" not in args.model.lower():
            if "Question" in orig_msg:
                msg = [{"role": "system", "content": ""},
                    {"role": "user", "content": orig_msg["Question"]}]
            elif "question" in orig_msg:
                msg = [{"role": "system", "content": ""},
                    {"role": "user", "content": orig_msg["question"]}]
            else:
                msg = [{"role": "system", "content": ""},
                    {"role": "user", "content": orig_msg["problem_text"]}]
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, 
                add_generation_prompt=True
            )
            chat_templated_texts.append(text)
            if "Correct Answer" in orig_msg:
                gold_answers.append(orig_msg["Correct Answer"])
            elif "ground_truth" in orig_msg:
                gold_answers.append(orig_msg["ground_truth"])
            else:
                gold_answers.append(orig_msg["answer_latex"])
        else:
            if "Question" in orig_msg:
                chat_templated_texts.append(orig_msg["Question"])
            elif "question" in orig_msg:
                chat_templated_texts.append(orig_msg["question"])
            else:
                chat_templated_texts.append(orig_msg["problem_text"])
            if "Correct Answer" in orig_msg:
                gold_answers.append(orig_msg["Correct Answer"])
            elif "ground_truth" in orig_msg:
                gold_answers.append(orig_msg["ground_truth"])
            else:
                gold_answers.append(orig_msg["answer_latex"])
    
    converted_data = data.add_column('texts', chat_templated_texts)
    converted_data = converted_data.add_column('answers', gold_answers)
    
    return converted_data

LLAMA3_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>\n\n' }}\n{% elif message['role'] == 'system' %}\n{{ bos_token + '\n\n<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>\n\n' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>\n\n' }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}\n{% endif %}\n{% endfor %}"

def inference_batch(args):
    logging.info("loading data and model...")

    sample_filename = f"results/{args.model.split('/')[-1]}.json"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    
    # add padding token if not already there
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    gen_configs = {
        "bos_token_id": 128000,
        "do_sample": True,
        "eos_token_id": [
            128001,
            128009
        ],
        "temperature": 0.6,
        "top_p": 0.9
    }

    generation_config = GenerationConfig.from_pretrained(
        args.model,
        max_new_tokens=1680,
        num_return_sequences=1,
        top_k=50,
        **gen_configs
    )

    # with open("data/eval/gpqa_main.csv", mode='r', encoding='utf-8') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     test_data = [row for row in csv_reader]

    test_data = list()
    if args.task == "gpqa":
        with open("data/eval/gpqa_main.json") as f:
            data = json.load(f)
        for d in data:
            if "no" in d["statement_question"]:
                test_data.append(d)
        
        test_data = Dataset.from_list(test_data)
    elif args.task == "scibench":
        test_data = load_dataset("xw27/scibench", split="train")
    
    elif args.task == 'math500':
        test_data = load_dataset('json', data_files="test_data/MATH_test500.json")["train"]
    
    # print("test_data[0]",test_data["question"][0])
    data = apply_chat_template(tokenizer, test_data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)

    final_outputs = list()
    with torch.inference_mode():
        s = 0
        for samples in tqdm(dataloader):
            instructions = samples['texts']
            answers = samples["answers"]
            input = tokenizer(instructions, return_tensors="pt", padding="longest", truncation=True)
            input_ids = input.input_ids.to(model.device)
            attention_mask = input.attention_mask.to(model.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)

            for instruction, output, answer in zip(instructions, outputs, answers):
                output_string = tokenizer.decode(output[input_ids.size(1):], skip_special_tokens=True)
                print(instruction)
                print()
                print("---")
                print()
                print(output_string)
                print()
                print("---")
                print()
                print(answer)
                print()
                print("=" * 50)
                print()

                final_outputs.append({"task": instruction, "output": output_string, "answer": answer})

            # s += 1
            # if s == 20:
            #     break
            
        os.makedirs(f"outputs_{args.task}",exist_ok=True)
        with open(f"outputs_{args.task}/{args.model.split('/')[-1]}.json", "w") as f:
            json.dump(final_outputs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--template", type=str, default='default')
    parser.add_argument("--save_folder", type=str, default="")
    parser.add_argument("--data_file", type=str, default="data/androidcontrol_test.json")
    parser.add_argument("--output_file_name", type=str, default=None)

    args = parser.parse_args()

    inference_batch(args)