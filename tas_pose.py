import pickle

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from dataset.assembly import Assembly101PosesDataModule,AssemblyDataModule
from model.poses import TemporalActionSegmentation
from model.utils.visualize import show_segm

d = Assembly101PosesDataModule(
    '/media/z1ko/2TM2/datasets/Assembly101',
    target='action', 
    batch_size=1
) 
d.setup()

#d = AssemblyDataModule(
#    'data/processed/assembly', 
#    batch_size=1,
#)
#d.setup()

model = TemporalActionSegmentation(
    model_dim=128,
    learning_rate=0.1,
    weight_decay=0.0005,
    scheduler_step=50,
    joint_count=2*21,
    joint_features=3,
    temporal_state_dim=256,
    spatial_state_dim=64,
    temporal_layers=4,
    spatial_layers=2,
    classes=202
)

logger = WandbLogger(save_dir='runs/', name='LORAS[pose,fine]')
trainer = Trainer(
    logger=logger,
    max_epochs=150,
    gradient_clip_val=300.0,
    accumulate_grad_batches=16,
)

trainer.fit(
    model, 
    train_dataloaders=d.train_dataloader(), 
    val_dataloaders=d.val_dataloader(),
)

scores = model.predict(d.validation)

with open('data/output/scores.pkl', 'wb') as f:
    pickle.dump(scores, f)

a = 0

#scores = sorted(scores, key=lambda x: x['mof'])
#visualize(scores[0]['results'], scores[0]['targets'], 25, 'data/output/best.png')
#visualize(scores[-1]['results'], scores[-1]['targets'], 25, 'data/output/worst.png')