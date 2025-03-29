import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import learn2learn as l2l
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler, TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import seed_everything
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AdamW,
)

from cs224u.compgen import recogs_exact_match
from helper import get_tokenizer, get_raw_dataset
from data import RecogsDataset
# from pytorch_lightning.loggers import WandbLogger  # 可选: 使用wandb进行实验跟踪
# import wandb

seed_everything(42, workers=True)


class T5RecogsModel(pl.LightningModule):
    def __init__(
        self,
        src_vocab_path,
        tgt_vocab_path,
        lr=1e-4,
        batch_size=64,
        num_workers=4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = T5Config.from_pretrained("google/flan-t5-base", tie_word_embeddings=False)
        self.config.tie_word_embeddings = False
        self.encdec = T5ForConditionalGeneration(self.config)
        self.encdec.shared = None
        self.enc_tok = get_tokenizer(src_vocab_path)
        self.dec_tok = get_tokenizer(tgt_vocab_path)
        self.raw_dataset = get_raw_dataset()
        # encoder_embedding = nn.Embedding(
        #     num_embeddings=self.enc_tok.vocab_size,
        #     embedding_dim=self.config.d_model,
        #     padding_idx=self.enc_tok.pad_token_id,
        # )
        # self.encdec.encoder.set_input_embeddings(encoder_embedding)
        # self.encdec.decoder.set_input_embeddings(self.encdec.shared)
        # self.encdec.encoder.resize_token_embeddings(self.enc_tok.vocab_size)
        # self.encdec.decoder.resize_token_embeddings(self.dec_tok.vocab_size)
        self.encdec.encoder.embed_tokens = nn.Embedding(
            self.enc_tok.vocab_size,
            self.config.d_model,
            padding_idx=self.enc_tok.pad_token_id,
            device=self.device,
        )
        self.encdec.decoder.embed_tokens = nn.Embedding(
            self.dec_tok.vocab_size,
            self.config.d_model,
            padding_idx=self.dec_tok.pad_token_id,
            device=self.device,
        )
        self.encdec.lm_head = nn.Linear(
            self.config.d_model,
            self.dec_tok.vocab_size,
            bias=False,
            device=self.device
        )
        self.encdec.lm_head.weight = self.encdec.decoder.embed_tokens.weight
        self.encdec.to(self.device)
        self.encdec.train()

    def forward(self, X_pad, X_mask, y_pad, y_mask):
        X_pad, X_mask, y_pad, y_mask = X_pad.to(self.device), X_mask.to(self.device), y_pad.to(self.device), y_mask.to(self.device)
        self.encdec.to(self.device)
        outputs = self.encdec(
            input_ids=X_pad,
            attention_mask=X_mask,
            decoder_attention_mask=y_mask,
            labels=y_pad
        )
        return outputs

    def training_step(self, batch, idx):
        X_pad, X_mask, y_pad, y_mask, label = [x.to(self.device) for x in batch]
        outputs = self(
            X_pad,
            X_mask,
            y_mask,
            label,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, idx):
        X_pad, X_mask, y_pad, y_mask, label = [x.to(self.device) for x in batch]
        self.validation_step_outputs.append({
            "preds": self._predict(X_pad, X_mask),
            "labels": label,
        })

    def on_validation_epoch_end(self):
        preds = []
        for x in self.validation_step_outputs:
            preds += x["preds"]
        labels = self.val_dataloader().dataset.original_y
        vals = [int(recogs_exact_match(gold, pred)) for gold, pred in zip(labels, preds)]
        accuracy = sum(vals) / len(vals)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        return {"val_accuracy": accuracy}

    def _predict(self, X_pad, X_mask):
        preds = []
        with torch.no_grad():
            # """
            # Mac临时代码
            # """
            # X_pad, X_mask = X_pad.to("cpu"), X_mask.to("cpu")
            # self.encdec.to("cpu")
            # """
            # END
            # """
            self.encdec.eval()

            X_pad, X_mask = X_pad.to(self.device), X_mask.to(self.device)
            outputs = self.encdec.generate(
                X_pad,
                attention_mask=X_mask,
                max_new_tokens=512,
                eos_token_id=self.encdec.config.eos_token_id
            )
            results = self.dec_tok.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            preds += results
        return preds

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=3,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        train_dataset = RecogsDataset(
            self.enc_tok,
            self.dec_tok,
            self.raw_dataset["train"].input,
            self.raw_dataset["train"].output,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dataset = RecogsDataset(
            self.enc_tok,
            self.dec_tok,
            self.raw_dataset["gen"].input,
            self.raw_dataset["gen"].output,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
        )


if __name__ == "__main__":
    # Early stopping mechanism
    config = {
        'batch_size': 64,
        'learning_rate': 1e-4,
        'max_epochs': 50,
        "device": "mps"
    }
    SRC_DIRNAME = "./data/recogs"
    model = T5RecogsModel(
        f"{SRC_DIRNAME}/src_vocab.txt",
        f"{SRC_DIRNAME}/tgt_vocab.txt",
        lr=config["learning_rate"],
        batch_size=config["batch_size"],
        num_workers=os.cpu_count() // 2,
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        min_delta=1e-6,
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='flan-t5-{epoch:02d}-{val_accuracy:.4f}',
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,  # Save Top 1 Model Weights
        save_last=False,
        save_weights_only=True,
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=config["device"],
        callbacks=[early_stopping, checkpoint_callback],
        deterministic=True,
        # precision="bf16-mixed"
    )
    trainer.fit(model)

