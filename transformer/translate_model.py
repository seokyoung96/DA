from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    MarianTokenizer,
    MarianMTModel,
    MarianConfig,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random


class CustomDataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def get_inputs(self, df):
        input_pairs = []

        for i in range(len(df)):
            input_text = df.iloc[i, 0]
            target_text = df.iloc[i, 1]

            input_ids = self.tokenizer(input_text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True)
            target_ids = self.tokenizer(target_text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True)


            input_pair = {'input_ids' : torch.LongTensor(input_ids['input_ids']),
                    'attention_mask' : torch.LongTensor(input_ids['attention_mask']),
                    'labels' : torch.LongTensor(target_ids['input_ids'])}

            input_pairs.append(input_pair)

        return input_pairs

    def get_input_ids(self, df):
        input_ids_list = []

        for i in range(len(df)):
            input_text = df.iloc[i, 0]
            target_text = df.iloc[i, 1]

            input_ids = self.tokenizer(input_text, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            input_ids_list.append(input_ids)

        return torch.tensor(input_ids_list)
    


class CustomModel(nn.Module):
    def __init__(self, custom_dataset, model_name, device, tokenizer, decoder_layers=6, dropout=None):
        super().__init__()

        self.custom_dataset = custom_dataset
        self.device = device
        self.tokenizer = tokenizer

        config = MarianConfig.from_pretrained(model_name)

        config.decoder_layers = decoder_layers

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config = config)

        if dropout:
            self.model.dropout = dropout
        # 수정

        # encoder freezing
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False

        self.batch_size = -1
        self.batch_cnt = -1

        self.test_result = [[],[]]

    def train(self, training_args, train_df, valid_df):
        train_dataset = self.custom_dataset.get_inputs(train_df)
        valid_dataset = self.custom_dataset.get_inputs(valid_df)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        trainer.train()

    def generate(self, df):
        input_ids = self.custom_dataset.get_input_ids(df)
        outputs = self.model.generate(input_ids.to(self.device), max_length=512)

        output_sentences = []
        for output in outputs:
            output_sentences.append(self.tokenizer.decode(output, skip_special_tokens=True))

        return output_sentences

    def test_accuracy(self, df, batch_size=100):
        self.batch_size = batch_size
        self.batch_cnt = 0
        self.test_result = [[], []]

        df_len = len(df)
        start,end = 0,self.batch_size

        performance = 0
        while 1:
            if start >= df_len:
                break

            end = min(end, df_len)
            performance += self._test_accuracy(df[start:end])

            start = end
            end += self.batch_size

        performance /= self.batch_cnt
        print(f'final performance : {performance}')

        return performance

    def _test_accuracy(self, df):
        self.batch_cnt += 1

        input_sentences = self.generate(df)
        target_sentences = [sentence for sentence in df.iloc[:, 1]]

        df_len = len(df)
        cnt_prev, cnt, cntO = ((self.batch_cnt-1) * self.batch_size),0,0

        print(f'### {self.batch_cnt} batch start ###')
        for i in range(df_len) :
            cnt += 1
            cnt_global = cnt_prev + cnt

            input_sentence = input_sentences[i]
            target_sentence = target_sentences[i]

            if input_sentence == target_sentence :
                cntO += 1
            else :
                self.test_result[0].append(input_sentence)
                self.test_result[1].append(target_sentence)

            # if cnt % 100 == 0:
            #     print(f'{cnt_global} generated')
        # if cnt % 100 != 0:
        #     print(f'{cnt_global} generated')

        performance = cntO/cnt
        print(f'{self.batch_cnt} batch performance : {performance}\n')

        return performance


    def return_model(self):
        return self.model