import torch
import torch.nn as nn
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)

from torch_model_base import TorchModelBase
from compgen import recogs_exact_match
from helper import SRC_DIRNAME, get_tokenizer, get_raw_dataset
from data import RecogsDataset


def set_seed(seed: int):
    """设置所有可能的随机种子"""
    import random
    import numpy as np
    import torch

    # Python随机种子
    random.seed(seed)
    # Numpy随机种子
    np.random.seed(seed)
    # PyTorch随机种子
    torch.manual_seed(seed)
    # CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        # 确保CUDA的运算是确定性的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 在代码开始处调用
set_seed(42)  # 使用固定的种子


def check_environment():
    """检查并打印环境信息"""
    import torch
    import platform
    import sys

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")

    # 检查计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computing device: {device}")

    # 检查数值精度
    print(f"Default dtype: {torch.get_default_dtype()}")

    return device


# 在训练开始前调用
device = check_environment()


class RecogsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "mean"

    def forward(self, outputs, labels):
        """`labels` is ignored, as it was already used to assign a
        value of `outputs.loss`, and that value is all we need."""
        return outputs.loss


class RecogsModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = T5Config.from_pretrained("google/byt5-base", tie_word_embeddings=False)
        self.config.tie_word_embeddings = False
        self.encdec = T5ForConditionalGeneration(self.config)
        self.encdec.shared = None
        # self.encdec = CustomEncoderDecoderModel.from_pretrained(
        #     f"ReCOGS/ReCOGS-model")
        self.encdec.encoder.embed_tokens = nn.Embedding(
            762,
            self.config.d_model,
            padding_idx=0,
        )
        self.encdec.decoder.embed_tokens = nn.Embedding(
            729,
            self.config.d_model,
            padding_idx=0,
        )
        self.encdec.lm_head = nn.Linear(
            self.config.d_model,
            729,
            bias=False,
        )
        self.encdec.lm_head.weight = self.encdec.decoder.embed_tokens.weight

    def forward(self, X_pad, X_mask, y_pad, y_mask, labels=None):
        outputs = self.encdec(
            input_ids=X_pad,
            attention_mask=X_mask,
            decoder_attention_mask=y_mask,
            labels=y_pad
        )
        return outputs


class RecogsModel(TorchModelBase):
    def __init__(self, *args,
            initialize=True,
            enc_vocab_filename=f"{SRC_DIRNAME}/src_vocab.txt",
            dec_vocab_filename=f"{SRC_DIRNAME}/tgt_vocab.txt",
            **kwargs):
        self.enc_vocab_filename = enc_vocab_filename
        self.dec_vocab_filename = dec_vocab_filename
        self.enc_tokenizer = get_tokenizer(self.enc_vocab_filename)
        self.dec_tokenizer = get_tokenizer(self.dec_vocab_filename)
        super().__init__(*args, **kwargs)
        self.loss = RecogsLoss()
        if initialize:
            self.initialize()

    def build_graph(self):
        return RecogsModule()

    def build_dataset(self, X, y=None):
        return RecogsDataset(
            self.enc_tokenizer, self.dec_tokenizer, X, y=y)

    def predict(self, X, device=None):
        device = self.device if device is None else torch.device(device)
        dataset = self.build_dataset(X)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        self.model.to(device)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_pad, X_mask = [x.to(device) for x in batch]
                outputs = self.model.encdec.generate(
                    X_pad,
                    attention_mask=X_mask,
                    max_new_tokens=512,
                    eos_token_id=self.model.encdec.config.eos_token_id)
                results = self.dec_tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                preds += results
        return preds

    def score(self, X, y, device=None):
        # An overall accuracy score:
        preds = self.predict(X, device=device)
        vals = [int(recogs_exact_match(gold, pred)) for gold, pred in zip(y, preds)]
        return sum(vals) / len(vals)


if __name__ == "__main__":
    recogs_model = RecogsModel(
        batch_size=16,
        max_iter=50,
        eta=1e-4,
        optimizer_class=torch.optim.AdamW,
        early_stopping=True,
        n_iter_no_change=5,
    )
    dataset = get_raw_dataset()
    # recogs_model.predict(dataset['dev'].input[: 2], device="cpu")
    recogs_model.fit(dataset["train"].input, dataset["train"].output)
    # gen = recogs_model.score(dataset["gen"].input, dataset["gen"].output)
    gen_result = recogs_model.score(dataset["gen"].input, dataset["gen"].output)
    # test_result = recogs_model.score(dataset["test"].input, dataset["test"].output, device="cpu")
    # print(f"Dev result: {dev_result}")
    print(f"Gen result: {gen_result}")
    torch.save(
        recogs_model.model.state_dict(),
        f"/content/drive/MyDrive/CS224U/checkpoints/flan-t5-{gen_result:.4f}.pth'"
    )
    # torch.save(
    #     recogs_model.model.state_dict(),
    #     f"./flan-t5-{1000:.4f}.pth"
    # )
    # print(f"Test result: {test_result}")