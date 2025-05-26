
def get_instruction(dataset_name, strategy, disorder_name):

    if disorder_name =='depression':
        disorder='Major Depressive Disorder'

    if strategy == 'no-cot':
        instruction = """Consider this post to answer the question: Is the poster likely to suffer from very severe {}? Only return Yes or No.""".format(disorder)
    elif strategy =='step-by-step' or strategy =='step-by-step_ours':
        instruction = 'Consider this post to answer the question: Is the poster likely to suffer from very severe {}? Only return Yes or No, then explain your reasoning step by step.'.format(disorder)
    elif strategy in ['std-cot', 'std-cot_ours', 'std-cot_worst', 'std-cot_random']:
        instruction = 'Consider this post to answer the question: Is the poster likely to suffer from very severe {}? Only return Yes or No, then explain your reasoning.'.format(disorder)
    elif strategy == 'symptom':
        instruction = "Consider the symptoms  of the {} expressed from this post to answer the question: Is the poster likely to suffer from very severe {}? Only return Yes or No, then explain your reasoning step by step.".format(disorder,disorder)        
    elif strategy == 'emotion' or strategy == 'emotion_ours':
        instruction = "Consider the emotions expressed from this post to answer the question: Is the poster likely to suffer from very severe {}? Only return Yes or No, then explain your reasoning step by step.".format(disorder)

    return instruction


def make_prompt(post, instruction):
    return '{}\nPost:"{}".\nAnswer: '.format(instruction, post)

