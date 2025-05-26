# openai
from openai import OpenAI
import numpy as np
import time, argparse, json
from tqdm import tqdm
from utils import compute_metrics, print_result, get_references, get_ref_sent_embd
from models import get_generative_model, request


import re, os
import csv


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./results/{}/k_samples/{}')
    parser.add_argument('--output_dir', type=str, default='./results/{}/selection_strategy_analysis/{}')

    parser.add_argument('--generic_no', type=int, default=0)
    parser.add_argument('--specific_no', type=int, default=0)
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--toy', type=int, default=0)

    # Model settings
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--target_model', type=str, default='gpt-4o')
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--api_key', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument('--save_point', type=int, default=100)

    # Instruction settings
    parser.add_argument('--strategy', type=str, default='emotion')
    #parser.add_argument('--strategy', type=str, default='step-by-step')
    #parser.add_argument('--strategy', type=str, default='std-cot')

    
    parser.add_argument('--disorder', type=str, default='depression')
    parser.add_argument('--dataset_name', type=str, default='Reddit_depression')
    parser.add_argument('--instruction_k', type=int, default=5)
    parser.add_argument('--temperature', type=int, default=0)
    parser.add_argument('--max_token_len', type=int, default=200)



    return parser.parse_args()


def make_gpt_eval_prompt(disorder_name, post, res):

    items = [
        "Depressed mood most of the day, nearly every day.",
        "Markedly diminished interest or pleasure in all, or almost all, activities most of the day, nearly every day.",
        "Insomnia or hypersomnia nearly every day.",
        "Significant weight loss when not dieting or weight gain, or decrease or increase in appetite nearly every day.",
        "Fatigue or loss of energy nearly every day.",
        "Feeling worthlessness or excessive or inappropriate guilt nearly every day.",
        "Diminished ability to think or concentrate, or indecisiveness, nearly every day.",
        "A slowing down of thought and a reduction of physical movement.",
        "Recurrent thoughts of death, recurrent suicidal ideation without a specific plan, or a suicide attempt or a specific plan for committing suicide."
    ]


    instruction = """You are a clinical psychologist with expertise in diagnosing Major Depressive Disorder (MDD). 
Your task is to score a reasoning text about potential MDD cases based on symptom identification and logical analysis. Return only a score without rationale.


Symptom Checklist
{}

Reasoning Quality Scale
1-2: SEVERELY INADEQUATE

No reference to diagnostic criteria
Unclear or disconnected descriptions
Primarily subjective interpretations

3-4: INADEQUATE

Mentions symptoms without connecting to criteria
Some relevant points but significant gaps
Overlooks several key symptoms

5-6: BASIC

Basic references from diagnostic criteria
Relevant but not insightful observations
Makes logical connections but lacks depth

7-8: PROFICIENT

Alignment with diagnostic criteria
Relevant and accurate with minor omissions
Cover key symptoms

9-10: EXEMPLARY

Clear alignment with diagnostic criteria
Highly relevant and comprehensive
Consider both key and minor symptoms with clear evidence

Input Format:
Post: [Original text]
Reasoning: [Analysis to evaluate]

Output Format:
Score: [1-10]

Post: "{}"
Reasoning: "{}"
""".format('\n'.join(items), post, res)


    return instruction



def extract_score(res):
    answers = re.findall(r'\b[0-9]\b', res)

    if len(answers) == 0:
        return 0
    return int(answers[0])




if __name__ == "__main__":
    args = get_args()
    print(args)

    with open(args.data_dir.format(args.dataset_name,'{}_{}_{}_top-10_{}.json'.format(args.mode, args.target_model, args.strategy, args.api_key)), 'r') as fp:
        print(args.data_dir.format(args.dataset_name,'{}_{}_{}_top-10_{}.json'.format(args.mode, args.target_model, args.strategy, args.api_key)))
        datas = json.load(fp)
        
    print(args.output_dir.format(args.dataset_name, '{}_{}_{}_selection_{}.json'.format(args.mode, args.model, args.strategy, args.api_key)))

    exact_matching_phrases = []
    root_path = './references/depression/patterns/'
    files = os.listdir(root_path)
    for file in files:
        with open(root_path+file, 'r') as fp:
            lines = fp.readlines()
            lines = [line.strip() for line in lines]
            print(len(lines))
            exact_matching_phrases = exact_matching_phrases+lines



    client=get_generative_model(args)


    result = dict()

    all_bleus = []

    save_idx = 0

    for idx, data in enumerate(tqdm(datas, desc='evaluating', mininterval=0.01, leave=True)):




        post = datas[data][0]
        label = datas[data][1]
        ress = datas[data][2]
        prompt = datas[data][3]
        data_id = data

        freqs = []
        freqs_post = []

        lengths = []

        gpt_evals = []
        gpt_scores = []

        if label == 1:

            for res in ress:
                # token length
                lengths.append(len(res.split(' ')))

                # exact matching
                freq = 0
                for phrase in exact_matching_phrases:
                    if re.search(r'\b' + phrase + r'\b', res):
                        freq+=1
                freqs.append(freq)

                freq = 0
                for phrase in exact_matching_phrases:
                    if re.search(r'\b' + phrase + r'\b', post):
                        freq+=1
                freqs_post.append(freq)

                # gpt-eval
                
                message = make_gpt_eval_prompt(args.disorder,post,res)
                answer = request(args, data_id, client, message)
                if answer==None:
                    continue
            
                score = extract_score(answer)
                gpt_evals.append(answer)
                gpt_scores.append(score)
        


        result[data_id] = [post, label, ress, prompt, lengths, freqs, freqs_post, gpt_scores, gpt_evals]
            
        if save_idx%args.save_point==0:
            with open(args.output_dir.format(args.dataset_name, '{}_{}_{}_selection_{}.json'.format(args.mode, args.target_model, args.strategy, args.api_key)), 'w') as fp:
                json.dump(result, fp, indent=4, sort_keys=False, ensure_ascii=False)
        save_idx+=1

        
    with open(args.output_dir.format(args.dataset_name, '{}_{}_{}_selection_{}.json'.format(args.mode, args.target_model, args.strategy, args.api_key)), 'w') as fp:
        json.dump(result, fp, indent=4, sort_keys=False, ensure_ascii=False)
        

        