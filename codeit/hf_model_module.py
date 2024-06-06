# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from typing import Any, Dict, Mapping, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from transformers import T5Config

from codeit.typing_custom import TensorType
from codeit.utils import get_class, get_tokenizer


def get_model(cls, name, cache_dir):
    model_class = get_class(cls)
    return model_class.from_pretrained(name, cache_dir=cache_dir)


class HFModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.transformer = None
        self.cls = config.model.cls
        if not os.path.isdir(config.model.resume_path):
            print(f"***{config.model.resume_path} does not exist so initialising base model ***")
            self.name = config.model.name
        else:
            print("*** model resuming from checkpoint ***")
            self.name = config.model.resume_path
        self.cache_dir = config.model.cache_dir
        self.reduce_size = config.model.reduce_size
        self.config = config
        self.save_hyperparameters(config)
        self.scheduler_kwargs = self.hparams["optimization"]["scheduler"]["kwargs"]
        self.random_weights = config.model.random_weights
        self.tokenizer = get_tokenizer(config=config)

    def setup(self, stage=None):
        if self.transformer == None:
            self.load_pretrained()

    def load_pretrained(self):
        print(f"loading model from hugging face {self.name}")
        model_class = get_class(self.cls)
        if self.random_weights:
            config = T5Config.from_pretrained(self.name)
            self.transformer = model_class(config)
        else:
            self.transformer = get_model(self.cls, self.name, self.cache_dir)
        if bool(self.reduce_size):
            my_config = self.transformer.config
            my_config.num_decoder_layers = 1
            my_config.num_heads = 1
            my_config.num_layers = 1
            my_config.d_model = 32
            try:
                self.transformer = model_class.from_config(my_config)
            except:
                self.transformer = model_class(my_config)

    def forward(
        self,
        input_ids: TensorType,
        attention_mask: TensorType,
        labels: Optional[TensorType],
        **kwargs: Any,
    ) -> Any:
        model_outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return model_outputs

    def training_step(self, batch: Dict[str, TensorType], batch_idx: int) -> STEP_OUTPUT:
        training_loss = self.step(batch, batch_idx)
        self.log(
            "train/loss", training_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return training_loss

    def validation_step(
        self, batch: Dict[str, TensorType], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        validation_loss = self.step(batch, batch_idx)
        self.log(
            "val/loss", validation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return validation_loss

    def configure_optimizers(self) -> Mapping[str, Any]:
        optimizer_class = get_class(self.hparams["optimization"]["optimizer"]["type"])
        optimizer = optimizer_class(
            self.parameters(), **self.hparams["optimization"]["optimizer"]["kwargs"]
        )
        if self.hparams["optimization"]["scheduler"]["cls"] != "None":
            if "num_training_steps" in self.scheduler_kwargs.keys():
                self.trainer.fit_loop.setup_data()
                steps_per_epoch = len(self.trainer.train_dataloader)
                num_train_steps_from_num_epochs = (
                    self.hparams["trainer"]["max_epochs"] * steps_per_epoch
                )
                if self.hparams["trainer"]["max_steps"] == -1:
                    self.hparams["optimization"]["scheduler"]["kwargs"][
                        "num_training_steps"
                    ] = num_train_steps_from_num_epochs
                else:
                    self.hparams["optimization"]["scheduler"]["kwargs"]["num_training_steps"] = min(
                        self.hparams["trainer"]["max_steps"], num_train_steps_from_num_epochs
                    )

            scheduler_class = get_class(self.hparams["optimization"]["scheduler"]["cls"])
            scheduler = scheduler_class(
                optimizer, **self.hparams["optimization"]["scheduler"]["kwargs"]
            )
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval=self.hparams["optimization"]["scheduler"]["interval"],
                    frequency=1,
                ),
            )
        else:
            return dict(optimizer=optimizer)

    def step(self, batch: Dict[str, TensorType], batch_idx: int) -> STEP_OUTPUT:
        outputs = self(**batch)
        loss = outputs.loss

        return loss
