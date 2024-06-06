# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools
import json

import datasets
import lightning.pytorch as pl
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader

from codeit.policy.tokenize import create_dataset, tokenize_simple_seq_2_seq


def get_inference_dataset(
    tasks,
    tokenizer,
    input_state_max,
    n_examples,
    max_decoder_tokens=512,
    sparse=True,
    text_encoder=None,
):
    inference_dataset = create_dataset(
        tasks,
        n_examples=n_examples,
        sparse=sparse,
        text_encoder=text_encoder,
    )
    inference_dataset = inference_dataset.map(
        tokenize_simple_seq_2_seq,
        fn_kwargs={
            "tokenizer": tokenizer,
            "input_state_max": input_state_max,
            "max_tokens": max_decoder_tokens,
        },
    )
    return inference_dataset


def empty_dataset():
    return datasets.Dataset.from_dict(
        {"input_ids": [], "attention_mask": [], "labels": [], "task_ids": []}
    )


def collate_fn_seq2seq(batch, pad_token_id=0):
    # padding to longest sequences in batch
    input_ids_padded = pad_sequence(
        [torch.tensor(batch_item["input_ids"]) for batch_item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_masks_padded = pad_sequence(
        [torch.tensor(batch_item["attention_mask"]) for batch_item in batch],
        batch_first=True,
        padding_value=0,
    )
    task_ids_padded = pad_sequence(
        [torch.tensor(batch_item["task_id"]) for batch_item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    if batch[0]["labels"][0]:
        labels_padded = pad_sequence(
            [torch.tensor(batch_item["labels"]) for batch_item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
    else:
        labels_padded = None
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_masks_padded,
        "task_id": task_ids_padded,
    }


class BucketSampler(BatchSampler):
    def __init__(
        self, data_source, sort_lens, batch_size, shuffle=True, drop_last=False, sampler=None
    ):
        self.sort_lens = sort_lens
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        super().__init__(data_source, batch_size, drop_last)

        if not shuffle:
            self.batches = self.create_batches()

    def create_batches(self):
        sorted_indices = np.argsort(self.sort_lens)
        batches = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]
        return batches

    def __iter__(self):
        if self.shuffle:
            self.batches = self.create_batches()
            np.random.shuffle(self.batches)
        return iter([[int(index) for index in batch] for batch in self.batches])

    def __len__(self):
        if self.drop_last:
            return len(self.sort_lens) // self.batch_size
        else:
            return (len(self.sort_lens) + self.batch_size - 1) // self.batch_size


class ExItDataModule(pl.LightningDataModule):
    def __init__(self, replay_buffer, config):
        super().__init__()
        self.config = config

        self.setup_called = False
        self.replay_buffer = replay_buffer

        self.collate_fn = functools.partial(
            collate_fn_seq2seq, pad_token_id=config.models.data.dataloader.tokenizer.pad_token_id
        )

        self.batch_size = config.data.dataloader.batch_size
        self.n_workers = config.data.dataloader.num_workers

        self.n_policy_experiences = self.config.replay_buffer.num_policy_experiences
        self.n_mutated_experiences = self.config.replay_buffer.num_mutated_experiences

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        if not self.setup_called:
            self.train_dataset = datasets.Dataset.from_dict(
                self.replay_buffer.sample_experiences(
                    n_policy_experiences=self.n_policy_experiences,
                    n_mutated_experiences=self.n_mutated_experiences,
                )
            )
            self.val_dataset = empty_dataset()
            self.test_dataset = empty_dataset()
            self.setup_called = True

    def train_dataloader(self):
        sort_lens = [len(x["labels"]) for x in self.train_dataset]
        bucket_sampler = BucketSampler(
            data_source=self.train_dataset,
            sort_lens=sort_lens,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        sort_lens = [len(x["labels"]) for x in self.val_dataset]
        bucket_sampler = BucketSampler(
            data_source=self.val_dataset,
            sort_lens=sort_lens,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        sort_lens = [len(x["labels"]) for x in self.test_dataset]
        bucket_sampler = BucketSampler(
            data_source=self.test_dataset,
            sort_lens=sort_lens,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.n_workers,
            shuffle=False,
        )
