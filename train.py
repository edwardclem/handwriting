from hw.models.crnn import CRNN
from hw.data import make_vocab, collate_fn
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


WANDB_PROJECT = "ctc"

ds = load_dataset("Teklia/IAM-line")
vocab = make_vocab(ds["train"])

collate_fn_wrapped = partial(collate_fn, vocab=vocab)
train_loader = DataLoader(
    ds["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn_wrapped,
    num_workers=3,
)
val_loader = DataLoader(
    ds["validation"],
    batch_size=10,
    shuffle=False,
    collate_fn=collate_fn_wrapped,
    num_workers=3,
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=WandbLogger(
        project=WANDB_PROJECT, dir="wandb", save_dir="wandb", log_model="all"
    ),
    max_epochs=500,
    callbacks=[
        ModelCheckpoint(monitor="val_loss", save_top_k=5),
        # EarlyStopping(monitor="val_loss"),
    ],
)

model = CRNN(vocab)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
