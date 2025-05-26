
import time, argparse, json
from tqdm import tqdm
from prompts import get_instruction, make_prompt

from vllm import LLM, SamplingParams
from models import get_generative_model, request



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./dataset/{}/{}')
    parser.add_argument('--output_dir', type=str, default='./results/{}/{}/{}/{}')


    # student model lists
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
    #parser.add_argument('--strategy', type=str, default='emotion')
    #parser.add_argument('--strategy', type=str, default='std-cot')


    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--api_key', type=int, default=0)
    parser.add_argument('--save_point', type=int, default=100)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpu_id', type=str, default='0')


    # Instruction settings
    parser.add_argument('--disorder', type=str, default='depression')
    parser.add_argument('--dataset_name', type=str, default='Reddit_depression')
    parser.add_argument('--instruction_k', type=int, default=5)
    parser.add_argument('--temperature', type=int, default=0)

    parser.add_argument('--toy', type=int, default=0)


    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    client=get_generative_model(args)
    instruction = get_instruction(args.dataset_name, args.strategy, args.disorder)

    # get data
    with open(args.data_dir.format(args.dataset_name,'{}.json'.format(args.mode)), 'r') as fp:
        datas = json.load(fp)


    result = dict()
    for idx, data in enumerate(tqdm(datas, desc='{}'.format(args.mode), mininterval=0.01, leave=True)):
        data_id = data
        post = datas[data][0]
        label = datas[data][1]
        
        if len(post.split(' ')) >=4000:
            continue
        
        message = make_prompt(post,instruction)
        res = request(args, data_id, client, message)
        
        #import IPython; IPython.embed(); exit(1)
        if res==None:
            continue
        result[data_id] = [post, label, res, message] 
        
        if idx%args.save_point==0:
            with open(args.output_dir.format(args.dataset_name, args.mode, args.teacher_model, '{}_{}.json'.format(args.model, args.strategy)), 'w') as fp:
                json.dump(result, fp, indent=4, sort_keys=False, ensure_ascii=False)
        
    with open(args.output_dir.format(args.dataset_name, args.mode, args.teacher_model, '{}_{}.json'.format(args.model, args.strategy)), 'w') as fp:
        json.dump(result, fp, indent=4, sort_keys=False, ensure_ascii=False)

#import IPython; IPython.embed(); exit(1)
