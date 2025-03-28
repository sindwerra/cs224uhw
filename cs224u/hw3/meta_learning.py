import pytorch_lightning as pl
import torch
import torch.nn as nn
import learn2learn as l2l
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger  # 可选: 使用wandb进行实验跟踪
# import wandb







class MetaReCOGSLightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        args: Dict
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        # 初始化MAML模型
        self.maml = l2l.algorithms.MAML(
            model,
            lr=args['inner_lr'],
            first_order=False,
            allow_unused=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录最佳验证损失
        self.best_val_loss = float('inf')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.maml.parameters(),
            lr=self.args['outer_lr']
        )
        
        # 可选：学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

    def compute_loss(self, batch, model):
        inputs = batch['input']
        targets = batch['output']
        
        outputs = model(inputs)
        loss = self.criterion(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        """单个训练步骤"""
        meta_train_loss = 0.0
        
        # 对批次中的每个任务进行训练
        for task_batch in batch:
            # 克隆学习器
            learner = self.maml.clone()
            
            # 划分支持集和查询集
            support_data = {
                k: v[:self.args['k_shot']] 
                for k, v in task_batch.items()
            }
            query_data = {
                k: v[self.args['k_shot']:] 
                for k, v in task_batch.items()
            }
            
            # 内循环适应
            for _ in range(self.args['adaptation_steps']):
                support_loss = self.compute_loss(support_data, learner)
                learner.adapt(support_loss)
            
            # 在查询集上计算损失
            query_loss = self.compute_loss(query_data, learner)
            meta_train_loss += query_loss
        
        meta_train_loss = meta_train_loss / len(batch)
        
        # 记录指标
        self.log('train_loss', meta_train_loss, prog_bar=True)
        
        return meta_train_loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        meta_val_loss = 0.0
        
        for task_batch in batch:
            learner = self.maml.clone()
            
            # 支持集适应
            support_data = {
                k: v[:self.args['k_shot']] 
                for k, v in task_batch.items()
            }
            query_data = {
                k: v[self.args['k_shot']:] 
                for k, v in task_batch.items()
            }
            
            for _ in range(self.args['adaptation_steps']):
                support_loss = self.compute_loss(support_data, learner)
                learner.adapt(support_loss)
            
            # 查询集评估
            query_loss = self.compute_loss(query_data, learner)
            meta_val_loss += query_loss
        
        meta_val_loss = meta_val_loss / len(batch)
        
        # 记录验证指标
        self.log('val_loss', meta_val_loss, prog_bar=True)
        
        return meta_val_loss

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.validation_step(batch, batch_idx)

class MetaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        args: Dict
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.args = args

    def setup(self, stage: Optional[str] = None):
        # 创建任务转换器
        self.task_transforms = [
            l2l.data.transforms.NWays(
                self.train_dataset,
                n=self.args['n_way']
            ),
            l2l.data.transforms.KShots(
                self.train_dataset,
                k=self.args['k_shot'] + self.args['k_query']
            ),
        ]

    def train_dataloader(self):
        train_tasks = l2l.data.TaskDataset(
            self.train_dataset,
            task_transforms=self.task_transforms,
            num_tasks=self.args['tasks_per_epoch']
        )
        return DataLoader(
            train_tasks,
            batch_size=self.args['batch_size'],
            shuffle=True,
            num_workers=self.args['num_workers']
        )

    def val_dataloader(self):
        val_tasks = l2l.data.TaskDataset(
            self.val_dataset,
            task_transforms=self.task_transforms,
            num_tasks=self.args['eval_tasks']
        )
        return DataLoader(
            val_tasks,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers']
        )

    def test_dataloader(self):
        test_tasks = l2l.data.TaskDataset(
            self.test_dataset,
            task_transforms=self.task_transforms,
            num_tasks=self.args['eval_tasks']
        )
        return DataLoader(
            test_tasks,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers']
        )

def main():
    # 配置参数
    args = {
        'vocab_size': 1000,
        'embedding_dim': 256,
        'hidden_dim': 512,
        'inner_lr': 0.01,
        'outer_lr': 0.001,
        'epochs': 50,
        'k_shot': 5,
        'k_query': 15,
        'n_way': 5,
        'tasks_per_epoch': 100,
        'eval_tasks': 50,
        'adaptation_steps': 5,
        'batch_size': 4,
        'num_workers': 4,
    }

    # 初始化模型
    base_model = ReCOGSModel(
        args['vocab_size'],
        args['embedding_dim'],
        args['hidden_dim']
    )
    
    # 初始化Lightning模块
    model = MetaReCOGSLightning(base_model, args)
    
    # 初始化数据模块
    # 注意：这里需要实际的数据集
    data_module = MetaDataModule(
        train_dataset=ReCOGSDataset(...),
        val_dataset=ReCOGSDataset(...),
        test_dataset=ReCOGSDataset(...),
        args=args
    )
    
    # 设置回调
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='meta-recogs-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # 设置logger
    # wandb_logger = WandbLogger(
    #     project='meta-recogs',
    #     name='experiment-1',
    #     log_model=True
    # )
    
    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=args['epochs'],
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )
    
    # 开始训练
    trainer.fit(model, data_module)
    
    # 测试模型
    trainer.test(model, data_module)


if __name__ == '__main__':
    main()
