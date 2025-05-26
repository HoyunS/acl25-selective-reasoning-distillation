
# utils
import numpy as np
import time, argparse, json
from tqdm import tqdm
from prompts import get_instruction, make_prompt
from utils import compute_metrics, print_result

import re, os
import csv


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_dir', type=str, default='./results/{}/{}/{}/{}')


    # Model settings
    # Model settings
    #parser.add_argument('--model', type=str, default='llama-3')
    parser.add_argument('--model', type=str, default='llama-2')
    #parser.add_argument('--model', type=str, default='mistral')


    # teacher model lists
    #parser.add_argument('--teacher_model', type=str, default='gpt35')
    #parser.add_argument('--teacher_model', type=str, default='gpt-4o')
    #parser.add_argument('--teacher_model', type=str, default='Llama-70b')
    parser.add_argument('--teacher_model', type=str, default='None')


    # strategy lists
    parser.add_argument('--strategy', type=str, default='step-by-step')
    #parser.add_argument('--strategy', type=str, default='no-cot')
    #parser.add_argument('--strategy', type=str, default='std-cot')


    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument("--eps", type=float, default=1e-6)

    # Instruction settings
    parser.add_argument('--disorder', type=str, default='depression')
    parser.add_argument('--dataset_name', type=str, default='Reddit_depression')
    parser.add_argument('--instruction_k', type=int, default=5)



    return parser.parse_args()



def extract_answer(res):
    answers = re.findall(r'(?<!\/)\b(Yes|No)\b(?!\/)', res)

    if len(answers) ==0:
        return 2
    elif answers[0]=='Yes':
        return 1
    elif answers[0]=='No':
        return 0
    return 2






if __name__ == "__main__":
    args = get_args()
    print(args)

    with open(args.target_dir.format(args.dataset_name, args.mode, args.teacher_model, '{}_{}.json'.format(args.model, args.strategy)), 'r') as fp:
        datas = json.load(fp)

    
    all_labels = []
    all_preds = []
    

    all_pos = []
    all_neg = []

    cnt_no_answer = 0

    for idx, data in enumerate(tqdm(datas, desc='evaluating', mininterval=0.01, leave=True)):
        post = datas[data][0]
        label = datas[data][1]
        res = datas[data][2]
        prompt = datas[data][3]
        

        pred = extract_answer(res)
        if pred==2:
            cnt_no_answer+=1
            pred=0

        all_preds.append(pred)
        
        
        all_labels.append(label)



    
    test_result = compute_metrics(labels=all_labels, preds=all_preds)
    print_result(test_result)
    print('original data size: ',len(datas))
    print('no answer percentage: ',cnt_no_answer, (cnt_no_answer/len(datas)))
    
        
    #import IPython; IPython.embed(); exit(1)

        