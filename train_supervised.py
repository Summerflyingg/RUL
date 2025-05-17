import pytorch_lightning as pl
from datasets.cmapss_dm import CMAPSSDataModule
from lit_modules.finetune_lit import FineTuneLit
import torch

def build_adj():
    import numpy as np
    np.random.seed(0)
    dummy = np.random.randn(1000,14)
    corr = np.corrcoef(dummy, rowvar=False)
    adj = (np.abs(corr) > 0.2).astype('float32')
    return torch.tensor(adj)

if __name__ == '__main__':
    ckpt = 'ssl_ckpt.ckpt'  # replace with real path
    dm = CMAPSSDataModule(data_root='data', batch_size=32, stage='supervised')
    model = FineTuneLit(adj=build_adj(), encoder_ckpt=ckpt)
    trainer = pl.Trainer(max_epochs=5, accelerator='cpu', limit_train_batches=5)
    trainer.fit(model, dm)
