from functools import partial
from typing import List, Dict, Any, Callable, Tuple

import lightning as L
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.transforms import v2


# make vocab with blank char as 0
# blank token encoding from https://pytorch.org/audio/main/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html#tokens
def make_vocab(dataset: List[Dict[str, Any]]) -> Dict[str, Dict[Any, Any]]:
    forward_vocab = {"<blk>": 0}
    for entry in dataset:
        for character in entry["text"]:
            if character not in forward_vocab:
                forward_vocab[character] = len(forward_vocab)
    inverse_vocab = {idx: char for char, idx in forward_vocab.items()}
    return {"forward": forward_vocab, "reverse": inverse_vocab}


def collate_fn(
    batch: List[Dict[str, Any]],
    vocab: Dict[str, Dict[Any, Any]],
    xforms: List[Callable],
    target_size: Tuple[int, int] = (128, 1000),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:

    # convert all to same size

    base_xforms = []
    if xforms:
        base_xforms += xforms
    base_xforms += [
        v2.Resize(target_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    pipeline = v2.Compose(base_xforms)
    padded_images = torch.stack([pipeline(entry["image"]) for entry in batch])

    string_seqs = [entry["text"] for entry in batch]

    # convert string sequence to toks
    unpadded_sequences = [
        torch.tensor([vocab["forward"][char] for char in string])
        for string in string_seqs
    ]

    # Pad sequences to the maximum length
    sequence_lengths = torch.tensor(
        [seq.shape[0] for seq in unpadded_sequences], dtype=torch.long
    )

    # concat all sequences!

    padded_sequences = torch.concat(unpadded_sequences)

    # padded_sequences = pad_sequence(
    #     unpadded_sequences, batch_first=True, padding_value=-1
    # )

    return padded_images, padded_sequences, sequence_lengths, string_seqs


class IAMLineDataModule(L.LightningDataModule):
    def __init__(self, train_xforms=None, batch_size=32, workers=3):
        super().__init__()
        self.train_xforms = train_xforms
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage: str):
        self.dataset = load_dataset("Teklia/IAM-line")
        self.vocab = make_vocab(self.dataset["train"])

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=32,
            shuffle=True,
            collate_fn=partial(collate_fn, vocab=self.vocab, xforms=self.train_xforms),
            num_workers=3,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=32,
            shuffle=False,
            collate_fn=partial(collate_fn, vocab=self.vocab, xforms=None),
            num_workers=3,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=32,
            shuffle=False,
            collate_fn=partial(collate_fn, vocab=self.vocab, xforms=None),
            num_workers=3,
        )
