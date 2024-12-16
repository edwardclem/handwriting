from time import time

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.transforms import v2

from hw.data import IAMLineDataModule
from hw.models.crnn import CRNN

xforms = [
    v2.RandomAffine(1),
    v2.RandomZoomOut(side_range=(1, 1.5)),
    v2.RandomPerspective(distortion_scale=0.1),
]

prefix = int(time())


datamodule = IAMLineDataModule(train_xforms=xforms, batch_size=32, workers=3)
# this is just b/c I have the vocab defined in the datamodule
datamodule.setup("")
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=TensorBoardLogger("tensorboard", version=prefix),
    max_epochs=1000,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            dirpath=f"artifacts/{prefix}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)

model = CRNN(vocab=datamodule.vocab)

# # load checkpoint - need to figure out how to save vocab properly here
# model = CRNN.load_from_checkpoint(
#     "artifacts/model-gynv7eg7:v47/model.ckpt", vocab=datamodule.vocab
# )


trainer.fit(model, datamodule=datamodule)
