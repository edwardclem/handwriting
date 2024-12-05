from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import v2

from hw.data import IAMLineDataModule
from hw.models.crnn import CRNN

WANDB_PROJECT = "ctc"


xforms = [
    v2.RandomAffine(10, (0.05, 0.05), (0.9, 1.0)),
    v2.RandomZoomOut(side_range=(1, 1.5)),
    v2.RandomPerspective(distortion_scale=0.1),
]

datamodule = IAMLineDataModule(train_xforms=xforms, batch_size=32, workers=3)
# this is just b/c I have the vocab defined in the datamodule
datamodule.setup("")
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=WandbLogger(
        project=WANDB_PROJECT, dir="wandb", save_dir="wandb", log_model="all"
    ),
    max_epochs=1000,
    callbacks=[
        ModelCheckpoint(monitor="val_loss", save_top_k=5, save_last=True),
    ],
)

model = CRNN(datamodule.vocab)

trainer.fit(model, datamodule=datamodule)
