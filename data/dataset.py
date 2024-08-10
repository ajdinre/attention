from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import BertTokenizer


class WMTDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
        subset_fraction: float = 1.0,
        vocab_size: int = 10000,
    ):
        self.dataset = load_dataset("wmt/wmt14", "de-en", split=split)

        if subset_fraction < 1.0:
            self.dataset = self.dataset.select(
                range(int(len(self.dataset) * subset_fraction))
            )

        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased",
            model_max_length=max_length,
            vocab_size=vocab_size,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.dataset[idx]
        source = item["translation"]["de"]
        target = item["translation"]["en"]

        # tokenize and encode the source and target
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "source_ids": source_encoding["input_ids"].squeeze(),
            "source_mask": source_encoding["attention_mask"].squeeze(),
            "target_ids": target_encoding["input_ids"].squeeze(),
            "target_mask": target_encoding["attention_mask"].squeeze(),
        }
