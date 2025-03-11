import json

import torch
from sklearn.metrics import classification_report

from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch.nn as nn


class ClassifierModule(nn.Module):
    def __init__(
        self,
        model_name,
        classifier_mode="pooler",
        activation="swish",
        p=0.1
    ) -> None:
        super().__init__()
        # This is specific sentiment analysis task, just hard code for 3 classes if fine
        self.n_classes = 3 
        self.core_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.n_classes)
        self.core_model.train()
        self.classifier_mode = classifier_mode
        self.hidden_dim = self.core_model.config.hidden_size
        self.p = p # Dropout probability

        # Use this as the classifier layer's activation function choice flag: Swish, Relu, Gelu
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU(inplace=True)

        # Use this as the final classifier choice: MaxPooling, AvgPooling, Logits
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4), # Follow standard transformer design
            self.activation,
            nn.Dropout(p=self.p),
            nn.Linear(self.hidden_dim * 4, self.n_classes),
            nn.Dropout(p=self.p)
        )

    def forward(self, x, attention_mask):
        x = self.core_model(x, attention_mask=attention_mask, output_hidden_states=True)
        if self.classifier_mode == "max_pooling":
            x = torch.max(x.hidden_states[-1], dim=1).values
            out = self.classifier(x)
        elif self.classifier_mode == "avg_pooling":
            x = torch.mean(x.hidden_states[-1], dim=1)
            out = self.classifier(x)
        else:
            out = x.logits
        return out


class Classifier(TorchShallowNeuralClassifier):
    def __init__(self, model_name, classifier_mode, activation_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.classifier_mode = classifier_mode
        self.activation = activation_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.params += ["model_name"]

    def build_graph(self):
        return ClassifierModule(
            model_name=self.model_name,
            classifier_mode=self.classifier_mode,
            activation=self.activation,
        )

    def build_dataset(self, X, y=None):
        data = get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'])
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)
        return dataset


###############################################Utility Functions Below##################################################

def get_batch_token_ids(batch, tokenizer):
   """Map `batch` to a tensor of ids. The return
   value should meet the following specification:

   1. The max length should be 512.
   2. Examples longer than the max length should be truncated
   3. Examples should be padded to the max length for the batch.
   4. The special [CLS] should be added to the start and the special
      token [SEP] should be added to the end.
   5. The attention mask should be returned
   6. The return value of each component should be a tensor.

   Parameters
   ----------
   batch: list of str
   tokenizer: Hugging Face tokenizer

   Returns
   -------
   dict with at least "input_ids" and "attention_mask" as keys,
   each with Tensor values

   """
   be = tokenizer.batch_encode_plus(
      batch,
      truncation=True,
      max_length=512,
      padding=True,
      add_special_tokens=True,
      return_attention_mask=True,
      return_tensors="pt",
   )
   return {
      "input_ids": be.input_ids,
      "attention_mask": be.attention_mask,
   }


def train(
    model_names,
    dataset,
    lr=1e-5,
    batch_size=32,
    n_iter_no_changes=5
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Current Device {device}")
    classifier_modes = ["max_pooling", "avg_pooling", "logits"]
    activation_names = ["relu", "gelu", "swish"]
    ds_name = dataset["name"]
    print(f"Current Dataset is {ds_name}")
    ds = load_dataset(dataset["path"], dataset["name"], trust_remote_code=True)
    experiment_results = []
    for model_name in model_names:
        print(f"Current Model is {model_name}")
        for cls_mode in classifier_modes:
            print(f"Current Classifier Mode is {cls_mode}")
            for hidden_activation in activation_names:
                print(f"Current Activation Choice is {hidden_activation}")
                classifier = Classifier(
                    model_name=model_name,
                    classifier_mode=cls_mode,
                    activation_name=hidden_activation,
                    hidden_activation=hidden_activation,
                    eta=lr,
                    batch_size=batch_size,
                    early_stopping=True,
                    n_iter_no_change=n_iter_no_changes,
                    optimizer_class=torch.optim.AdamW,
                    l2_strength=1e-2,
                    device=device,
                    max_iter=2
                )
                mdl = classifier.fit(
                    ds["train"]["sentence"],
                    ds["train"]["gold_label"],
                )
                print("\nClassification Report Generation...")
                preds = classifier.predict(ds["validation"]["sentence"])
                report = classification_report(ds["validation"]["gold_label"], preds, digits=5)
                experiment_results.append({
                    "model_name": model_name,
                    "classifier_mode": cls_mode,
                    "activation": hidden_activation,
                    "dataset": dataset,
                    "device": device,
                    "best_error": mdl.best_error,
                    "best_score": mdl.best_score,
                    "validation_score": mdl.validation_scores,
                    "validation_report": report
                })
                with open(f"{model_name}-{cls_mode}-{hidden_activation}-{dataset}.json", "w") as f:
                    json.dump(results, f)
                # mdl.best_parameters 是模型参数，后面要用
                # torch.save(mdl.best_parameters, f"{model_name}-{cls_mode}-{hidden_activation}-{}.pth")
            # FIXME 把数据集确定
    return experiment_results


if __name__ == "__main__":
    results = train(
        model_names=[
            "ProsusAI/finbert",
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "google/electra-base-discriminator",
            "BAAI/bge-reranker-v2-m3",
            "j-hartmann/emotion-english-distilroberta-base"
        ],
        dataset={"path": "dynabench/dynasent", "name": "dynabench.dynasent.r2.all"},
        batch_size=64,
    )

