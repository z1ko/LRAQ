import pickle

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from dataset.assembly import AssemblyDataModule
from model.tas import TemporalActionSegmentation
from model.utils.visualize import visualize

d = AssemblyDataModule(
    'data/processed/assembly', 
    batch_size=6,
    window_size=2000
)
d.setup()

model = TemporalActionSegmentation(
    model_dim=64,
    learning_rate=0.0001,
    weight_decay=0.0005,
    scheduler_step=30,
    joint_count=2*21,
    joint_features=3,
    temporal_state_dim=128,
    spatial_state_dim=32,
    temporal_layers=8,
    spatial_layers=2
)

logger = WandbLogger(save_dir='runs/', name='LRGA_TAS_POSE')
trainer = Trainer(max_epochs=20, logger=logger)

trainer.fit(
    model, 
    train_dataloaders=d.train_dataloader(), 
    val_dataloaders=d.val_dataloader()
)

scores = model.predict(d.validation)

with open('data/output/scores.pkl', 'wb') as f:
    pickle.dump(scores, f)

a = 0

#scores = sorted(scores, key=lambda x: x['mof'])
#visualize(scores[0]['results'], scores[0]['targets'], 25, 'data/output/best.png')
#visualize(scores[-1]['results'], scores[-1]['targets'], 25, 'data/output/worst.png')