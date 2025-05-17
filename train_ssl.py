import pytorch_lightning as pl
from lit_modules.ssl_lit import SSLLit
from datasets.cmapss_dm import CMAPSSDataModule
import torch

def build_adj():
    # simple Pearson adjacency on random synthetic data for placeholder
    import numpy as np
    np.random.seed(0)
    dummy = np.random.randn(1000,14)
    corr = np.corrcoef(dummy, rowvar=False)
    adj = (np.abs(corr) > 0.2).astype('float32')
    return torch.tensor(adj)

if __name__ == '__main__':
    dm = CMAPSSDataModule(data_root='data', batch_size=32, stage='ssl')
    adj = build_adj()
    model = SSLLit(adj=adj)
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu', limit_train_batches=5)
    trainer.fit(model, dm)
