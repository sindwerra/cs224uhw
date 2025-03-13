import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoModel, AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from datasets import load_dataset
from collections import Counter
from typing import Dict, List, Optional

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


class ExpertModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.dense(hidden_states)))
        x = self.layernorm(x)
        x = self.out_proj(x)
        return x


class SentimentClassifier(pl.LightningModule):
    def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-base",
            num_experts: int = 5,
            num_labels: int = 3,
            hidden_size: int = 768,
            dropout: float = 0.4,
            learning_rate: float = 2e-5
    ):
        super().__init__()
        self.save_hyperparameters()  # 这个直接把上面的所有args都保存了，也就是lr也保存起来了，这个特性实在不是很好...

        # 加载预训练模型
        self.roberta = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.deberta.config.hidden_dropout_prob = dropout
        self.deberta.config.attention_probs_dropout_prob = dropout

        # 专家模型
        self.experts = nn.ModuleList([
            ExpertModel(hidden_size, hidden_size // 2, num_labels, dropout)
            for _ in range(num_experts)
        ])

        # 路由层
        self.router = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )

        # # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(num_labels + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )

        # 类别权重
        self.register_buffer('class_weights', torch.ones(num_labels))
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids, attention_mask)

        # 获取[CLS]表示
        cls_output = outputs.last_hidden_state[:, 0]

        # 路由决策
        router_weights = self.router(cls_output)

        # 专家输出
        expert_outputs = torch.stack([
            expert(cls_output) for expert in self.experts
        ]).permute(1, 0, 2)
        expert_output = torch.sum(
            router_weights.unsqueeze(-1) * expert_outputs,
            dim=1
        )

        # 合并专家输出和pooler输出
        pooler_output = outputs.pooler_output
        final_input = torch.cat([expert_output, pooler_output], dim=-1)
        logits = self.classifier(final_input)

        return logits, router_weights

    def compute_loss(self, logits, labels, router_weights):
        # 分类损失
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        # 路由损失
        entropy = -(router_weights * torch.log(router_weights + 1e-10)).sum(-1).mean()

        # L2正则化
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param)

        return ce_loss + 0.1 * entropy + 0.01 * l2_reg

    def training_step(self, batch, batch_idx):
        logits, router_weights = self(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(logits, batch['labels'], router_weights)

        preds = torch.argmax(logits, dim=1)
        f1 = self.compute_f1(preds, batch['labels'])

        self.log_dict({
            'train_loss': loss,
            'train_f1': f1
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, router_weights = self(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(logits, batch['labels'], router_weights)

        preds = torch.argmax(logits, dim=1)
        self.validation_step_outputs.append({
            'val_loss': loss,
            'preds': preds,
            'labels': batch['labels']
        })
        self.log('val_loss_step', loss, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])

        f1 = self.compute_f1(all_preds, all_labels)

        self.log_dict({
            'val_loss': avg_loss,
            'val_f1': f1
        }, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        return {'val_loss': avg_loss, 'val_f1': f1}

    def compute_f1(self, preds, labels):
        return torch.tensor(f1_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            average='macro'
        ))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=3,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1',
                'frequency': 1
            }
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def setup(self, stage):
        # 加载数据集
        dynasent_r1 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all", trust_remote_code=True)
        dynasent_r2 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r2.all", trust_remote_code=True)
        sst5 = load_dataset("SetFit/sst5", trust_remote_code=True)

        dataset_dict = {
            "r1": dynasent_r1,
            "r2": dynasent_r2,
            "sst": sst5,
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
        'model_name': "cardiffnlp/twitter-roberta-base-sentiment-latest",
        'batch_size': 64,
        'max_length': 512,
        'num_experts': 5,
        'learning_rate': 2e-5,
        'max_epochs': 50
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
    model = SentimentClassifier(
        model_name=config['model_name'],
        num_experts=config['num_experts'],
        learning_rate=config['learning_rate']
    )

    # 设置类别权重
    model.class_weights = data_module.class_weights

    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=6,
        min_delta=1e-6,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',  # 保存路径
        filename='RoBERTa-sentiment-{epoch:02d}-{val_f1:.4f}',  # 文件名格式
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
