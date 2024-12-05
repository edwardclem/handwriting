from functools import partial

from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from hw.data import collate_fn, make_vocab
from hw.models.crnn import CRNN

WANDB_PROJECT = "ctc"

ds = load_dataset("Teklia/IAM-line")
vocab = make_vocab(ds["train"])

xforms = [
    v2.RandomAffine(10, (0.05, 0.05), (0.9, 1.0)),
    v2.RandomZoomOut(side_range=(1, 1.5)),
    v2.RandomPerspective(distortion_scale=0.1),
]

collate_train = partial(collate_fn, vocab=vocab, xforms=xforms)
train_loader = DataLoader(
    ds["train"],
    batch_size=32,
    shuffle=True,
    collate_fn=collate_train,
    num_workers=3,
)
collate_val = partial(collate_fn, vocab=vocab, xforms=None)
val_loader = DataLoader(
    ds["validation"],
    batch_size=32,
    shuffle=False,
    collate_fn=collate_val,
    num_workers=3,
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=WandbLogger(project=WANDB_PROJECT, dir="wandb", save_dir="wandb"),
    max_epochs=1000,
    callbacks=[
        ModelCheckpoint(monitor="val_loss", save_top_k=5),
        # EarlyStopping(monitor="val_loss"),
    ],
)

model = CRNN(vocab)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
