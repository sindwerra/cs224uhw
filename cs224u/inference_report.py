import json
import torch
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from datasets import load_dataset

dynar2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all', trust_remote_code=True)
dynar1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all', trust_remote_code=True)
sst = load_dataset("SetFit/sst5", trust_remote_code=True)

ds_names = ["r1", "r2", "sst"]


def convert_sst_label(s):
    return s.split(" ")[-1]


for splitname in ('train', 'validation', 'test'):
    dist = [convert_sst_label(s) for s in sst[splitname]['label_text']]
    sst[splitname] = sst[splitname].add_column('gold_label', dist)
    sst[splitname] = sst[splitname].add_column('sentence', sst[splitname]['text'])


ds = [dynar1, dynar2, sst]


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


def generate_deberta_report(weight_path):
    from hw1_bakeoff import SentimentClassifier
    model_name = "microsoft/deberta-v3-base"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentimentClassifier.load_from_checkpoint(weight_path)
    model.eval()
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    reports = []
    for i in range(len(ds)):
        d = ds[i]
        for name in ('validation', 'test'):
            X, y = d[name]["sentence"], d[name]["gold_label"]

            data = get_batch_token_ids(X, tokenizer)
            classes_ = sorted(set(y))
            n_classes_ = len(classes_)
            class2index = dict(zip(classes_, range(n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)

            if hasattr(dataset, "collate_fn"):
                collate_fn = dataset.collate_fn
            else:
                collate_fn = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn)
            preds = []
            labels = []
            model.to(device)
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, attetion_mask, gold_label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logits, _ = model(input_ids, attention_mask=attetion_mask)
                    preds.append(logits)
                    labels.append(gold_label)

            if all(x.shape[1: ] == preds[0].shape[1: ] for x in preds[1: ]):
                preds = torch.cat(preds, axis=0)
            else:
                preds = [p for batch in preds for p in batch]
                
            labels = torch.cat(labels, axis=0)
            probs = torch.softmax(preds, dim=1).cpu().numpy()
            final_preds = probs.argmax(axis=1)
            raw_result = classification_report(labels.cpu().numpy(), final_preds, digits=5, output_dict=True)
            raw_result.update({ds_names[i]: {name: raw_result}})

            reports.append(raw_result)
    return reports


def generate_electra_report(weight_path):
    from hw1_bakeoff_electrac import SentimentClassifier
    model_name = "google/electra-base-discriminator"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentimentClassifier.load_from_checkpoint(weight_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reports = []
    for i in range(len(ds)):
        d = ds[i]
        for name in ('validation', 'test'):
            X, y = d[name]["sentence"], d[name]["gold_label"]

            data = get_batch_token_ids(X, tokenizer)
            classes_ = sorted(set(y))
            n_classes_ = len(classes_)
            class2index = dict(zip(classes_, range(n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)

            if hasattr(dataset, "collate_fn"):
                collate_fn = dataset.collate_fn
            else:
                collate_fn = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn)
            preds = []
            labels = []
            model.to(device)
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, attetion_mask, gold_label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logits, _ = model(input_ids, attention_mask=attetion_mask)
                    preds.append(logits)
                    labels.append(gold_label)

            if all(x.shape[1: ] == preds[0].shape[1: ] for x in preds[1: ]):
                preds = torch.cat(preds, axis=0)
            else:
                preds = [p for batch in preds for p in batch]
                
            labels = torch.cat(labels, axis=0)
            probs = torch.softmax(preds, dim=1).cpu().numpy()
            final_preds = probs.argmax(axis=1)
            raw_result = classification_report(labels.cpu().numpy(), final_preds, digits=5, output_dict=True)
            reports.append({ds_names[i]: {name: raw_result}})
    return reports


if __name__ == "__main__":
    # generate_roberta_report()
    model_results = {}
    model_results["deberta"] = generate_deberta_report("/Users/duli/Downloads/deberta-sentiment-epoch=00-val_f1=0.7903.ckpt")
    model_results["electra"] = generate_electra_report("/Users/duli/Downloads/electra-sentiment-epoch=01-val_f1=0.7886.ckpt")
    rows = []
    
    # 遍历每个模型的结果
    for model_name, results in model_results.items():
        # 遍历每个类别的指标
        for result in results:
            for ds_name, value in result.items():
                for group_name, val in value.items():
                    for class_label, metrics in val.items():
                        row = {
                            'model': model_name,
                            'class': class_label,
                            "ds_name": ds_name,
                            "group_name": group_name,
                            # **metrics  # 展开metrics字典
                        }
                        if class_label in "accuracy":
                            row.update({"accuracy": metrics})
                        else:
                            row.update(**metrics)
                        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    df.to_excel("inference_result.xlsx")
    
