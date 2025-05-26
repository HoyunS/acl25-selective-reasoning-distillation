import time, argparse, json
from tqdm import tqdm
from prompts import get_instruction, make_prompt
from vllm import LLM, SamplingParams
import openai

import os


def get_generative_model(args):
    '''
    :param      args: get args with openai/google API key
    :return:    openai client (obj)
    '''
    import os
    os.environ ['CUDA_LAUNCH_BLOCKING'] = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    model = None

    if args.model == 'llama-3':
        model_name = 'models/Llama-3.1-8B-Instruct'
    elif args.model == 'mistral':
        model_name = 'models/Mistral-7B-Instruct-v0.1'
    elif args.model == 'llama-2':
        model_name = 'models/Llama-2-7b-chat-hf'



    # if use trained student model
    if args.teacher_model != 'None':
        peft_model_dir = './models/{}/{}/{}/'.format(args.teacher_model, args.model, args.strategy)
        merged_model_dir = './models/{}/{}/{}/merged/'.format(args.teacher_model, args.model, args.strategy)

        if not os.path.exists(peft_model_dir):
            os.makedirs(peft_model_dir)
        if not os.path.exists(merged_model_dir):
            os.makedirs(merged_model_dir)



        # merge adapter to model
        if len(os.listdir(merged_model_dir)) == 0:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            print('***merging model')
            peft_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_dir)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(merged_model_dir)

            print('***loading tokenizer model')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(merged_model_dir+"/tokenizer/")

        print('***loading tokenizer model')
        model=LLM(model=merged_model_dir, tokenizer=merged_model_dir+"/tokenizer/") 
        # breakpoint()

    else:
        if args.model == 'gpt35' or args.model =='gpt-4o':   # gpt-3.5-turbo, gpt-4o
            api_key= "<securely_stored_api_key>"
            model = openai.OpenAI(api_key=api_key)
        elif args.model=='Llama-70b':
            api_key= "<securely_stored_api_key>"
            model = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepinfra.com/v1/openai"
                )
        else:
            model = LLM(model=model_name)

    return model



def generate_openai_model(args, data_id, client, messages):
    cnt=0
    response = None
    if args.model == 'gpt35':
        model_name = 'gpt-3.5-turbo-1106'
    elif args.model == 'gpt-4o':
        model_name = 'gpt-4o'


    while True:
        try:
            cnt+=1
            if cnt==5:
                break
            response = client.chat.completions.create(
                model=model_name,
                temperature=args.temperature,
                messages=messages
            )
            break
            
        except Exception as e:
            print("Exception: ", e)
            print("Id: ", data_id)
            time.sleep(10)

    if response == None:
        return None
    res = response.choices[0].message.content
    return res


def request(args, data_id, client, message): 
    if args.model == 'gpt35':
        messages = [
            {"role": "user", "content": message},
        ]
    elif args.model == 'gpt-4o':
        messages = [
            {"role": "system", "content": 'You are an assistant to mental health professionals, your role is to support them by providing responses that help in identifying potential mental health issues. You must provide your own opinion in each response, even when the information available is limited. If you detect any indications of mental health concerns, please respond with "Yes".'},
            {"role": "user", "content": message},
        ]
    elif args.model == 'Llama-70b':
        messages = [
            {"role": "user", "content": message},
        ]


    if args.model == 'gpt35' or args.model=='gpt-4o' or args.model=='Llama-70b':
        res = generate_openai_model(args, data_id, client, messages)

    else:
        sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95,  max_tokens=300)
        output = client.generate(message, sampling_params)
        res = output[0].outputs[0].text
    return res
