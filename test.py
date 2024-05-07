import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary

from deepant import AnomalyDetector, DataModule, TrafficDataset, DeepAnt

from utils import plot_predictions, loss_plot, ts_plot

pl.seed_everything(42, workers=True)


if not torch.cuda.is_available():
    print('Please Activate GPU Accelerator if available')
else:
    print('Everything is Set')

df = pd.read_csv(r'C:\Users\82102\Desktop\daycare\DeepAnT\TravelTime_451.csv', index_col='timestamp', parse_dates=['timestamp'])
df.plot(figsize=(15, 6), title='Travel Time', legend=False)

SEQ_LEN = 10
dataset = TrafficDataset(df, SEQ_LEN)
target_idx = dataset.timestamp # Timestamps to detect where the Anomaly Happens
X, y = dataset[0]
X.shape, y.shape, len(dataset) # Checking Sizes are compatible...

model = DeepAnt(SEQ_LEN, 1)
sample = torch.randn(32,1,10)

model(sample).shape

model = DeepAnt(SEQ_LEN, 1)
anomaly_detector = AnomalyDetector(model)
dm = DataModule(df, SEQ_LEN)
mc = ModelCheckpoint(
    dirpath = 'checkpoints',
    save_last = True,
    save_top_k = 1,
    verbose = True,
    monitor = 'train_loss',
    mode = 'min'
    )

mc.CHECKPOINT_NAME_LAST = f'DeepAnt-best-checkpoint'
summary(model)

trainer = pl.Trainer(max_epochs=30,
                    accelerator="gpu",
                    devices=1,
                    callbacks=[mc],
                    #progress_bar_refresh_rate=30,
                    #fast_dev_run=True,
                    #overfit_batches=1
                    )
trainer.fit(anomaly_detector, dm)

anomaly_detector = AnomalyDetector.load_from_checkpoint('checkpoints/DeepAnt-best-checkpoint.ckpt',
                                model = model)

output = trainer.predict(anomaly_detector, dm)
preds_losses = pd.Series(torch.tensor([item[1] for item in output]).numpy(), index = target_idx)

THRESHOLD = 0.5
plot_predictions(preds_losses, THRESHOLD)

loss_plot(preds_losses, THRESHOLD)

print('Anomalies Detected: ')
preds_losses.loc[lambda x: x > THRESHOLD]

ts_plot(df, preds_losses, THRESHOLD)


ts_plot(df, preds_losses, THRESHOLD, range = ('2015-07-28', '2015-08-15'))