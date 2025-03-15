import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import DebertaV2Model, DebertaV2Tokenizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
import numpy as np
from datasets import load_dataset
from collections import Counter
from typing import Dict, List, Optional
from hw1_bakeoff import SentimentClassifier

torch.set_float32_matmul_precision('high')

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, dataset_name, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.data = self.process_dataset(dataset)
        self.label_dict = {"negative": 0, "neutral": 1, "positive": 2}

    def process_dataset(self, dataset) -> List[Dict]:
        if self.dataset_name != "sst":
            return [{"sentence": item["sentence"], "gold_label": item["gold_label"]} for item in dataset]

        processed_data = []

        def convert_sst_label(s):
            return s.split(" ")[-1]

        for item in dataset:
            # 统一输入格式
            dist = convert_sst_label(item["label_text"])
            processed_data.append({
                "sentence": item["text"],
                "gold_label": dist,
            })

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['sentence'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        if isinstance(item['gold_label'], list):
            print(item['gold_label'])
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label_dict[item['gold_label']])
        }

class SentimentDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-base",
            batch_size: int = 32,
            max_length: int = 512
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def setup(self, stage):
        # 加载数据集
        twitter_airline_ds = load_dataset("osanseviero/twitter-airline-sentiment", trust_remote_code=True)
        # dynasent_r2 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r2.all", trust_remote_code=True)
        # sst5 = load_dataset("SetFit/sst5", trust_remote_code=True)
        # amazon_review_ds =

        dataset_dict = {
            "twitter_airline": twitter_airline_ds,
            # "r2": dynasent_r2,
            # "sst": sst5,
        }

        self.train_dataset = ConcatDataset([
            SentimentDataset(
                dataset['train'],
                self.tokenizer,
                name,
                self.max_length
            ) for name, dataset in dataset_dict.items()
        ])

        self.val_dataset = ConcatDataset([
            SentimentDataset(
                dataset['validation'],
                self.tokenizer,
                name,
                self.max_length
            ) for name, dataset in dataset_dict.items()
        ])

        # 计算类别权重
        all_labels = [ent["labels"].item() for ent in self.train_dataset]
        label_counts = Counter(all_labels)
        max_count = max(label_counts.values())
        self.class_weights = torch.tensor([
            max_count / label_counts[i] for i in range(3)
        ])
        self.class_weights[1] += 1

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )


def train_model():
    # 配置参数
    config = {
        'model_name': "microsoft/deberta-v3-base",
        'batch_size': 64,
        'max_length': 512,
        'num_experts': 5,
        'learning_rate': 5e-6,
        'max_epochs': 10
    }

    # 初始化数据模块
    data_module = SentimentDataModule(
        model_name=config['model_name'],
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )

    # 准备数据
    data_module.setup("fit")

    # 初始化模型
    # model = SentimentClassifier(
    #     model_name=config['model_name'],
    #     num_experts=config['num_experts'],
    #     learning_rate=config['learning_rate']
    # )
    model = SentimentClassifier.load_from_checkpoint(
        "./deberta-sentiment-epoch=01-val_f1=0.7971.ckpt"
    )

    # 设置类别权重
    model.class_weights = data_module.class_weights

    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=3,
        min_delta=1e-6,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',  # 保存路径
        filename='deberta-sentiment-continue-{epoch:02d}-{val_f1:.4f}',  # 文件名格式
        monitor='val_f1',  # 监控的指标
        mode='max',  # 因为是f1分数所以用max
        save_top_k=1,  # 保存得分最高的k个模型
        save_last=True,  # 同时保存最后一个epoch的模型
        verbose=True
    )

    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        callbacks=[early_stopping, checkpoint_callback],
        precision="bf16-mixed",  # 使用混合精度训练
        gradient_clip_val=1.0
    )

    # 开始训练
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train_model()
