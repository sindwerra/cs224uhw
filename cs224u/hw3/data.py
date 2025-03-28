import torch
import pandas as pd


class RecogsDataset(torch.utils.data.Dataset):
    def __init__(self, enc_tokenizer, dec_tokenizer, X, y=None):
        self.X = [enc_tokenizer.encode(s) for s in X]
        self.y = y
        if y is not None:
            self.y = [dec_tokenizer.encode(s) for s in y]

    @staticmethod
    def collate_fn(batch):
        """Unfortunately, we can't pass the tokenizer in as an argument
        to this method, since it is a static method, so we need to do
        the work of creating the necessary attention masks."""
        def get_pad_and_mask(vals):
            lens = [len(i) for i in vals]
            maxlen = max(lens)
            pad = []
            mask = []
            for ex, length in zip(vals, lens):
                diff = maxlen - length
                pad.append(ex + ([0] * diff))
                mask.append(([1] * length) + ([0] * diff))
            return torch.tensor(pad), torch.tensor(mask)
        batch_elements = list(zip(*batch))
        X = batch_elements[0]
        X_pad, X_mask = get_pad_and_mask(X)
        if len(batch_elements) == 1:
            return X_pad, X_mask
        else:
            y = batch_elements[1]
            y_pad, y_mask = get_pad_and_mask(y)
            # Repeat `y_pad` because our optimizer expects to find
            # labels in final position. These will not be used because
            # Hugging Face will calculate the loss for us.
            return X_pad, X_mask, y_pad, y_mask, y_pad

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return (self.X[idx],)
        else:
            return (self.X[idx], self.y[idx])


def load_split(filename):
    return pd.read_csv(
        filename,
        delimiter="\t",
        names=['input', 'output', 'category'])

