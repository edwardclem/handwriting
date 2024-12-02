import torch
from torch.nn.utils.rnn import pad_sequence

import torchvision.transforms.functional as TF


# make vocab with blank char as 0
# blank token encoding from https://pytorch.org/audio/main/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html#tokens
def make_vocab(dataset):
    forward_vocab = {"<blk>": 0}
    for entry in dataset:
        for character in entry["text"]:
            if character not in forward_vocab:
                forward_vocab[character] = len(forward_vocab)
    inverse_vocab = {idx: char for char, idx in forward_vocab.items()}
    return {"forward": forward_vocab, "reverse": inverse_vocab}


def collate_fn(batch, vocab):
    """
    Collate function for batching (image, sequence) pairs with padding.

    Args:
        batch (list of dicts): A list of dicts.
            - image: A PIL image.
            - sequence: A 1D torch.Tensor of variable length.
        vocab

    Returns:
        padded_images (torch.Tensor): Batch of images padded to the maximum width, shape (N, C, H, max_W).
        padded_sequences (torch.Tensor): Batch of padded sequences, shape (N, max_seq_len).
        sequence_lengths (torch.Tensor): Original lengths of sequences in the batch.
    """
    # Separate images and sequences from the batch
    images = [TF.to_tensor(entry["image"]) for entry in batch]
    string_seqs = [entry["text"] for entry in batch]

    # Determine the maximum width of images
    max_width = max(img.shape[2] for img in images)

    # TODO: additional aug!!!

    # Pad images to the maximum width
    padded_images = torch.stack(
        [
            torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, 0))
            for img in images
        ]
    )

    # convert string sequence to toks
    unpadded_sequences = [
        torch.tensor([vocab["forward"][char] for char in string])
        for string in string_seqs
    ]

    # Pad sequences to the maximum length
    sequence_lengths = torch.tensor(
        [seq.shape[0] for seq in unpadded_sequences], dtype=torch.long
    )
    padded_sequences = pad_sequence(
        unpadded_sequences, batch_first=True, padding_value=0
    )

    return padded_images, padded_sequences, sequence_lengths, string_seqs
