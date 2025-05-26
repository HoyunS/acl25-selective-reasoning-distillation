import re
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, TaskType
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, auc, roc_curve)

trainerStage2datasetStage = {
    'tr': 'train',
    'val': 'validation',
    'test': 'test',
}


def pad_sequences(sequences, max_len, pad_value):
    return np.array(
        [np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=(pad_value,)) for seq in sequences])


def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def f1_pre_rec_scalar(all_labels, all_preds, main_label=1):
    labels = np.array(all_labels)
    preds = np.array(all_preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=main_label)#roc_curve(np.sort(labels), np.sort(preds), pos_label=main_label)
    recall = recall_score(labels, preds, average=None)
    if len(recall) == 1:
        main_label = 0
    return {
        "acc": simple_accuracy(labels, preds),
        #"precision_micro": precision_score(labels, preds, average="micro"),
        #"recall_micro": recall_score(labels, preds, average="micro"),
        #"f1_micro": f1_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average=None)[main_label],
        "precision": precision_score(labels, preds, average=None)[main_label],
        "f1": f1_score(labels, preds, average=None)[main_label],
        #"f1_macro": f1_score(labels, preds, average="macro"),
        "AUC": auc(fpr, tpr),
        # "recall_weighted": recall_score(labels, preds, average="weighted"),
        # "precision_weighted": precision_score(labels, preds, average="weighted"),
        # "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec_scalar(labels, preds)


def print_result(test_result):
    for name, value in test_result.items():
        print('   Average '+name, value)


def extract_answer(res):
    answers = re.findall(r'(?<!\/)\b(Yes|No)\b(?!\/)', res)

    if len(answers) == 0:
        return 2
    elif answers[0] == 'Yes':
        return 1
    elif answers[0] == 'No':
        return 0
    return 0


LORA_CONFIG = {
    '../models/Meta-Llama-3-8B-Instruct': LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ),
    '../models/Mistral-7B-Instruct-v0.1': LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ),
    '../models/Llama-2-7b-chat-hf': LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ),
}

