import torch
import torch.nn as nn
from transformers import EncoderDecoderModel

from cs224u.torch_model_base import TorchModelBase
from cs224u.compgen import recogs_exact_match
from helper import SRC_DIRNAME, get_tokenizer, get_raw_dataset
from data import RecogsDataset


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
        self.encdec = EncoderDecoderModel.from_pretrained(
            f"ReCOGS/ReCOGS-model")

    def forward(self, X_pad, X_mask, y_pad, y_mask, labels=None):
        outputs = self.encdec(
            input_ids=X_pad,
            attention_mask=X_mask,
            decoder_attention_mask=y_mask,
            labels=y_pad)
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
    recogs_model = RecogsModel()
    dataset = get_raw_dataset()
    # recogs_model.predict(dataset['dev'].input[: 2], device="cpu")
    recogs_model.fit(dataset["train"].input[:1000], dataset["train"].output[:1000])
    dev_result = recogs_model.score(dataset["dev"].input, dataset["dev"].output, device="cpu")
    # gen_result = recogs_model.score(dataset["gen"].input, dataset["gen"].output, device="cpu")
    test_result = recogs_model.score(dataset["test"].input, dataset["test"].output, device="cpu")
    print(f"Dev result: {dev_result}")
    # print(f"Gen result: {gen_result}")
    print(f"Test result: {test_result}")