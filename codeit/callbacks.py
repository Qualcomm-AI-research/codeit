# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import lightning.pytorch as pl
from fsspec.core import url_to_fs


class HfModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            trainer.lightning_module.transformer.save_pretrained(hf_save_dir)
            trainer.lightning_module.tokenizer.save_pretrained(hf_save_dir)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)
