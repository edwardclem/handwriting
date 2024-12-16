from time import time

from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.transforms import v2

from hw.data import IAMLineDataModule
from hw.models.crnn import CRNN

xforms = [
    v2.RandomAffine(5, translate=(0.05, 0.05), scale=(0.95, 1), shear=5),
    v2.RandomZoomOut(side_range=(1, 1.5)),
    v2.RandomPerspective(distortion_scale=0.1),
]

epoch = int(time())

datamodule = IAMLineDataModule(train_xforms=xforms, batch_size=32, workers=3)
# this is just b/c I have the vocab defined in the datamodule
datamodule.setup("")
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=TensorBoardLogger("tensorboard", version=epoch),
    max_epochs=1000,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            dirpath=f"artifacts/{epoch}",
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ],
)

model = CRNN(vocab=datamodule.vocab.tolist())

trainer.fit(model, datamodule=datamodule)
