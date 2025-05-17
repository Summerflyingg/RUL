# SelfVision-Lit

Re‑implementation of the TACDA pipeline with our new graph‑attention + time‑frequency PatchTST backbone, built on **PyTorch‑Lightning**.

## Quick start

```bash
# create venv + install deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Stage 0‑2: self‑supervised pre‑training
python train_ssl.py --config configs/ssl.yaml

# Stage 3: fine‑tune on source domain
python train_supervised.py --config configs/finetune.yaml

# Stage 4: domain adaptation (two‑step)
python train_da.py --config configs/da.yaml
```

All paths are managed with `OmegaConf` config files in **configs/** (omitted here for brevity).
