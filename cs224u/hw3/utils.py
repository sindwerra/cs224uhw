import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

SRC_DIRNAME = "../data/recogs"


def get_tokenizer(vocab_filename):
    with open(vocab_filename) as f:
        vocab = f.read().splitlines()
    vocab_size = len(vocab)
    vocab = dict(zip(vocab, list(range(vocab_size))))
    tok = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    # This definitely needs to be done here and in the construction of
    # `PreTrainedTokenizerFast`. Don't be tempted to "clean this up"!
    tok.add_special_tokens(["[BOS]", "[UNK]", "[PAD]", "[EOS]"])
    tok.pre_tokenizer = WhitespaceSplit()
    tok.post_processor = TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        special_tokens=[
            ("[BOS]", tok.token_to_id("[BOS]")),
            ("[EOS]", tok.token_to_id("[EOS]"))])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="[BOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        # This vital; otherwise any periods will have their leading
        # spaces removed, which is wrong for COGS/ReCOGS.
        clean_up_tokenization_spaces=False)


def load_split(filename):
    return pd.read_csv(
        filename,
        delimiter="\t",
        names=['input', 'output', 'category'])


def get_raw_dataset():
    dataset = {}

    for splitname in ("train", "dev", "test", "gen"):
        dataset[splitname] = load_split(f"{SRC_DIRNAME}/{splitname}.tsv")
    return dataset
