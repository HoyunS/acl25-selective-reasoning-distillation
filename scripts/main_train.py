import os
import re
import sys
import json
import math
import torch
import wandb
import random
import logging
import datetime
import evaluate
import transformers
import bitsandbytes
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from filelock import FileLock
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from einops import rearrange, repeat, reduce
from torch import einsum
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig,
)

from torchmetrics.text import SacreBLEUScore
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from peft import get_peft_model, PeftModel, PeftConfig, LoraConfig, prepare_model_for_kbit_training
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import utils


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SPLIT = ['train', 'validation', 'test']
RAW_DATAFNAME = ['train_zero.json', 'valid_zero.json', 'test_zero.json']

NEW_HPARAMS = {
    # 'generate_all_flags': False,
    # 'count_tgt_upper_max_length': 0,
}


class LoRAModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        if not self._should_skip_saving_checkpoint(trainer):
            pl_module.model.model.save_pretrained(f"{pl_module.hparams.output_dir}/{pl_module.hparams.model_name_or_path}/{trainer.logger.experiment.name}")
            pl_module.model.model.save_pretrained(os.path.join(f"{pl_module.hparams.output_dir}/{pl_module.hparams.model_name_or_path}/{trainer.logger.experiment.name}", str(trainer.current_epoch)))

        super().on_validation_end(trainer, pl_module)


class RedditDepressionDataset(Dataset):
    def __init__(self, cache_dir, data_dir, dataset_type, ratio, demo_dev_run):
        raw_fnames = [f'{data_dir}/{dataset_type}/{raw_fname}' for raw_fname in RAW_DATAFNAME]
        split_fnames = [f'{data_dir}/{dataset_type}_{split}_{ratio}.tsv' for split in SPLIT]

        if not all(os.path.exists(split_fname) for split_fname in split_fnames):
            self.prepare_split_datasets(raw_fnames, ratio, split_fnames)

        df_splits = {split: pd.read_csv(split_fname, sep='\t', header=0) for split, split_fname in zip(SPLIT, split_fnames)}
        if demo_dev_run:
            df_splits = {split: df_split[:200] for split, df_split in df_splits.items()}

        self.datasets = DatasetDict({split: Dataset.from_pandas(df_split) for split, df_split in df_splits.items()})

    def prepare_split_datasets(self, raw_fnames, ratio, split_fnames):
        df_result = {}
        for split, split_fname, raw_fname in zip(SPLIT, split_fnames, raw_fnames):
            df = pd.read_json(raw_fname)
            df = df.transpose()
            df = df.reset_index(drop=True)
            df['sample_index'] = range(len(df))
            df.to_csv(split_fname, sep='\t', index=False)
            df_result[split] = df

        return df_result

    def __len__(self, stage='train'):
        return len(self.datasets[stage])

    def __getitem__(self, idx, stage='train'):
        return self.datasets[stage][idx]

    def get_batch(self, indices, dataset_stage, input_column):
        indices = indices.tolist()  # Convert tensor to list
        return self.datasets[dataset_stage].select(indices)[input_column]


class LMModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Define a model class compatible with Hugging Face's transformers library
class LMModelForHF(PreTrainedModel):
    config_class = LMModelConfig

    def __init__(self, config, hparams):
        super().__init__(config)
        self.model = LMModel(hparams.model_name_or_path, hparams.cache_dir, hparams.use_lora_training, hparams.ckpt_fname)

    def forward(self, x):
        return self.model(x)

    def save_pretrained(self, save_directory):
        # Save the config
        self.config.save_pretrained(save_directory)
        # Save the model weights
        torch.save(self.model.state_dict(), f"{save_directory}/pytorch_model.bin")


class LMModel(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, use_lora_training, ckpt_fname):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        if use_lora_training:
            # setting LoRA config
            if ckpt_fname:
                self.model = PeftModel.from_pretrained(model=base_model, model_id=ckpt_fname)
            else:
                config = utils.LORA_CONFIG[model_name_or_path]
                self.model = get_peft_model(base_model, config)
            logger.info(f"[INFO] LoRA inference_mode: {self.model.peft_config['default'].inference_mode}")
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        logger.info(f"[INFO] {model_name_or_path} model loaded.")

    def forward(self, batch):
        outputs = self.model(**batch)
        loss = outputs[0]
        logits = outputs[1]

        return outputs, loss, logits


class DPCModule(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.root_model_name_or_path = self.hparams.model_name_or_path if not self.hparams.ckpt_fname else self.hparams.model_name_or_path
        self.model = LMModel(self.hparams.model_name_or_path, self.hparams.cache_dir, self.hparams.use_lora_training, self.hparams.ckpt_fname)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
            cache_dir=self.hparams.cache_dir,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        special_tokens_dict = {'pad_token': '[pad]'}  # 50257
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.model.resize_token_embeddings(len(self.tokenizer))

        self.model.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.model.config_pad_token_id = self.tokenizer.pad_token_id
        self.generation_config = GenerationConfig(
            max_length=self.hparams.max_seq_length,
            num_beams=self.hparams.num_beams,
            top_p=self.hparams.top_p,
            top_k=self.hparams.top_k,
            temperature=self.hparams.temperature,
            do_sample=False if math.isclose(self.hparams.top_p, 1.0, rel_tol=1e-6, abs_tol=1e-6) and
                               math.isclose(self.hparams.temperature, 1.0, rel_tol=1e-6, abs_tol=1e-6) and
                               self.hparams.num_beams == 1 and
                               self.hparams.top_k == 50 else True,
            eos_token_id=self.model.model.config.eos_token_id,
            pad_token_id=self.model.model.config.pad_token_id,
        )

        self.input_column = 'input'
        self.output_column = 'output'

        # self.metric_list = ['bertscore']
        self.input_keys = ['input_ids', 'attention_mask', 'labels']
        self.input_tensor_keys = ['input_ids', 'attention_mask', 'input_ids_eval', 'attention_mask_eval', 'labels']
        self.count_upper_max_length, self.count_tgt_upper_max_length = 0, 0

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def set_new_hparams(self):
        # for loading old lightningmodule checkpoint
        new_hparams = NEW_HPARAMS
        [setattr(self.hparams, k, v) for k, v in new_hparams.items() if k not in self.hparams]

    def setup(self, stage):
        datasets = RedditDepressionDataset(
            self.hparams.cache_dir,
            self.hparams.data_dir,
            self.hparams.dataset_type,
            self.hparams.ratio,
            self.hparams.demo_dev_run,
        ).datasets


        self.datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[],
            num_proc=None,  # default None
            load_from_cache_file=False,  # default False
            desc="Running tokenizer on dataset line_by_line",
        )

        logger.info(f'{self.hparams.task_name} dataset (over_len: {self.count_upper_max_length}/{self.count_tgt_upper_max_length}) loaded.')

    def tokenize_function(self, examples):
        inputs = examples[self.input_column]
        outputs = examples.get(self.output_column, [])

        batch_encoding = self.preprocess_texts(inputs, outputs)

        return batch_encoding

    def preprocess_texts(self, inputs, outputs):
        batch_encoding_eval = self.tokenizer(
            inputs,
            padding='max_length',
            truncation=True,
            max_length=self.hparams.max_src_length,
            return_tensors='np',
        )

        # Check if outputs are provided
        if outputs:
            combined_inputs_outputs = [x + y for x, y in zip(inputs, outputs)]
        else:
            combined_inputs_outputs = inputs

        batch_encoding = self.tokenizer(
            combined_inputs_outputs,
            padding='max_length',
            truncation=True,
            max_length=self.hparams.max_seq_length,
            return_tensors='np',
        )

        batch_encoding['labels'] = batch_encoding['input_ids'].copy()
        batch_encoding['input_ids_eval'] = batch_encoding_eval['input_ids'].copy()
        batch_encoding['attention_mask_eval'] = batch_encoding_eval['attention_mask'].copy()

        self.count_upper_max_length = self.count_upper_max_length + sum(
            1 for lst in batch_encoding.input_ids_eval if lst[0] != self.tokenizer.pad_token_id)
        self.count_tgt_upper_max_length = self.count_tgt_upper_max_length + sum(
            1 for lst in batch_encoding.input_ids if lst[0] != self.tokenizer.pad_token_id)

        return batch_encoding

    def _common_step(self, batch, batch_idx, stage):
        inputs = {k: v for (k, v) in batch.items() if k in self.input_keys}
        outputs, loss, logits = self.model(inputs)
        self.log(f'{stage}/loss', loss, batch_size=self.hparams.batch_size, sync_dist=True)

        if stage == 'tr':
            return loss

    def _common_epoch_end(self, outputs, stage):
        pass

    def _common_epoch_gather_end(self, outputs, stage):
        input_seqs, predictions, ref_labels, gathered_ids = [], [], [], []

        tensors_key = outputs[0].keys()
        dataset_stage = utils.trainerStage2datasetStage[stage]
        for j, output in enumerate(outputs):
            if (not self.hparams.do_train) and len(self.trainer.device_ids) >= 1:
                for key in tensors_key:
                    if output[key].numel() and len(output[key].shape) > 2:
                        output[key] = output[key].view(-1, output[key].shape[-1])
                    elif output[key].numel() and len(output[key].shape) == 2:
                        output[key] = output[key].view(-1)

            data_samples = self.datasets[dataset_stage].select(output["ids"].tolist())
            inputs, ref_label, pred_ids = data_samples[self.input_column], data_samples['label'], output["pred_ids"]

            pad_input_mask = torch.cat(
                (
                    output['attention_mask_eval'],
                    torch.zeros(
                        (pred_ids.shape[0],
                         max(0, pred_ids.shape[1] - output['attention_mask_eval'].shape[1])),
                        dtype=torch.int64,
                        device=self.device
                    )
                ), dim=1
            )
            pred_ids[pad_input_mask == 1] = self.tokenizer.pad_token_id

            pred_tokens = self.tokenizer.batch_decode(
                pred_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_tokens = [x.strip() for x in pred_tokens]

            input_seqs.extend(inputs)
            predictions.extend(pred_tokens)
            ref_labels.extend(ref_label)
            gathered_ids.extend(output["ids"].tolist())

        pred_labels = [utils.extract_answer(pred) for pred in predictions]
        total_result = self.evaluate_results(pred_labels=pred_labels, ref_labels=ref_labels)
        result_fname = os.path.join(self.hparams.output_dir, f'result_{stage}_{self.trainer.logger.experiment.name}.tsv')
        df = pd.DataFrame(self.datasets[dataset_stage][gathered_ids])
        df.drop(columns=self.input_tensor_keys, inplace=True)
        df["prediction"] = predictions
        df["pred_label"] = pred_labels
        df = df.replace(r'[\r\n]+', '  ', regex=True)

        logger.info("\n\n##### examples:\n\n")
        for i in range(0, min(self.hparams.batch_size, 5)):
            logger.info(f"input {i}: {df['input'][i]}")
            logger.info(f"pred  {i}: {df['prediction'][i]}")
            logger.info(f"pred_label {i}: {df['pred_label'][i]}")
            logger.info(f"label {i}: {df['label'][i]}")

        df.to_csv(result_fname, sep='\t', index=False)

        for k, v in total_result.items():
            self.log(f'{stage}/{k}', round(v, 4), sync_dist=True)

    def evaluate_results_deprecated(self, decoded_preds, decoded_labels):
        total_result = {}
        for metric_name in self.metric_list:
            metric = evaluate.load(metric_name)
            tmp_result, result = {}, {}
            if metric_name == 'bertscore':
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, model_type='distilbert-base-uncased')
                tmp_result['bertscore'] = sum(result['f1']) / len(result['f1'])
                result = tmp_result

            total_result.update(result)

        return total_result


    def evaluate_results(self, pred_labels, ref_labels):
        test_result = utils.compute_metrics(ref_labels, pred_labels)
        utils.print_result(test_result)
        return test_result

    def training_step(self, batch, batch_idx):
        outputs = self._common_step(batch, batch_idx, 'tr')
        self.test_step_outputs.append(outputs)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self._common_step(batch, batch_idx, 'val')
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._common_step(batch, batch_idx, 'test')
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        self.test_step_outputs = [output for output in self.test_step_outputs if type(output) == dict]
        self._common_epoch_end(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._common_epoch_end(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def on_training_epoch_end(self):
        if not self.hparams.overfit_batches > 0.0:
            pass
        else:
            self._common_epoch_end(self.training_step_outputs, 'tr')
            self.training_step_outputs.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and 'lora_' in n and not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and 'lora_' in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, betas=self.hparams.betas,
                          eps=self.hparams.eps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.get_estimated_stepping_batches(),
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.hparams.monitor,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            },
        }

    def get_estimated_stepping_batches(self):
        if self.hparams.max_steps and self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps * self.hparams.accumulate_grad_batches
        else:
            t_total = int(
                (
                        len(self.datasets['train'])
                        // (self.hparams.batch_size * max(1, len(self.hparams.devices)))
                )
                * self.hparams.max_epochs
                // self.hparams.accumulate_grad_batches
            )

        return t_total

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.datasets['validation'], self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, collate_fn=self._collate_fn)

    def _collate_fn(self, batch):
        df = pd.DataFrame(batch)
        input_dict = df.to_dict(orient='list')
        for key in self.input_tensor_keys:
            input_dict[key] = torch.tensor(input_dict[key])

        if 'sample_index' in input_dict:
            input_dict['sample_index'] = torch.tensor(input_dict['sample_index'])

        return input_dict


def get_checkpoint_callbacks(hparams):
    callbacks = []
    split, monitor_name = re.split('/', hparams.monitor)
    filename = 'epoch={epoch}-' + split + '_' + monitor_name
    checkpoint_class = LoRAModelCheckpoint if hparams.use_lora_training else ModelCheckpoint

    callbacks.append(
        checkpoint_class(
            dirpath=hparams.output_dir,
            filename=filename,
            monitor=hparams.monitor,
            save_last=True,
            save_top_k=1,
            mode=hparams.monitor_mode,
            verbose=True,
            auto_insert_metric_name=False
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval='step'))

    return callbacks


def main(hparams):
    seed_everything(hparams.seed, workers=True)

    params = vars(hparams)

    model = DPCModule(**params)
    model.set_new_hparams()

    checkpoint_callbacks = get_checkpoint_callbacks(hparams)
    logger_callback = TensorBoardLogger("dpc", name=hparams.task_name) if hparams.use_logger == 'tensorboard' \
        else WandbLogger(project=hparams.task_name)

    trainer = Trainer(
        devices=hparams.devices,
        accelerator=hparams.accelerator,
        strategy=hparams.strategy,
        max_epochs=hparams.max_epochs,
        max_steps=hparams.max_steps,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        overfit_batches=hparams.overfit_batches,
        gradient_clip_val=hparams.gradient_clip_val,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches,
        log_every_n_steps=hparams.log_every_n_steps,
        fast_dev_run=hparams.fast_dev_run,
        callbacks=checkpoint_callbacks,
        logger=logger_callback,
        deterministic=True,
    )

    if hparams.do_train:
        trainer.fit(model)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = ArgumentParser(add_help=False)

    # initialization
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_logger", default="wandb", type=str, choices=["tensorboard", "wandb"])
    parser.add_argument('--num_display', type=int, default=4)
    parser.add_argument('--monitor', type=str, default='val/loss')
    parser.add_argument('--monitor_mode', type=str, default='min')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--use_early_stopping', action='store_true', default=False)
    parser.add_argument('--use_lora_training', action='store_true', default=False)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--do_test", action="store_true", default=True)

    # dataset arguments
    parser.add_argument('--dataset_type', default='Reddit_depression', type=str, help='Reddit_depression')
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--cache_dir", default='cache', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--ratio", default=0.1, type=str)

    # model arguments
    parser.add_argument('--model_name_or_path', default='meta-llama/Llama-2-7b-hf')
    # parser.add_argument('--model_name_or_path', default='mistralai/Mistral-7B-Instruct-v0.1')

    parser.add_argument('--task_name', default='depression_classification', help="output dir name and project name in wandb")
    parser.add_argument('--tuning_type', default='fine-tuning', help="[fine-tuning]")
    parser.add_argument("--is_scratch", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument('--ckpt_fname', default='', type=str)

    parser.add_argument("--max_seq_length", default=768, type=int)
    parser.add_argument("--max_src_length", default=500, type=int)
    parser.add_argument("--max_tgt_length", default=268, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    ### training arguments (trainer)
    parser.add_argument("--accumulate_grad_batches", default=16, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)

    ###
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, default=(0.9, 0.98), nargs='+')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=16)

    # training arguments (trainer)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--strategy', default='ddp')
    parser.add_argument("--overfit_batches", default=0.0, type=float, help="use only #<v>/<v>% train batches")
    parser.add_argument("--gradient_clip_val", default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument("--limit_train_batches", default=1.0, type=float, help="How much of training dataset to check")
    parser.add_argument("--limit_val_batches", default=1.0, type=float, help="How much of training dataset to check")
    parser.add_argument("--limit_test_batches", default=1.0, type=float, help="How much of training dataset to check")
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--fast_dev_run', action='store_true', default=False)
    parser.add_argument('--demo_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()
    assert not (len(hparams.devices) != 1 and hparams.load_in_8bit), "8bit with multigpu will get divided device errors"

    logger.info("hparams used: {}".format(hparams))
    logger.info(torch.cuda.device_count())
    logger.info(torch.cuda.get_device_name(0))

    main(hparams)



