# NeurIPS Open Polymer Prediction 2025 — PolymerGNN (Silver, 49/2240)

Predicting five polymer properties (Tg, FFV, Tc, Density, Rg) from SMILES with a **graph neural network** and a **polymer‑aware pipeline**.  
This code finished **49 / 2240 teams (Silver medal)** on the private leaderboard.

> Competition: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025

---

## TL;DR
- **Encoder:** custom GNN with GRU message passing and attentive pooling; supports polymer edge modes (`pair`, `cycle`, `clique`).  
- **Targets:** multi‑task regression (5 properties) with **masked wMAE** (matches Kaggle metric). Optional **heteroscedastic head** that predicts per‑target σ.  
- **Self‑supervised pretraining (SSL):** InfoNCE on large unlabeled pool with node/edge drop augmentations. Saves `out_dir/ssl_encoder.pt` and can be reused.  
- **Data splits:** scaffold‑based split and **K‑fold by Murcko scaffolds** to reduce leakage; CV seeds supported.  
- **Stability tricks:** EMA, optional SWA, optional dot‑stage polishing.  
- **Inference:** TTA via MC‑Dropout, optional **winsorize/range‑match** calibration, optional **kNN‑blend** on SSL embeddings.  
- **Output:** `submission_*.csv` and blended `submission.csv` in `out_dir`.

## Repo layout
```
train_polymer_gnn_pro.py   # all-in-one training script (SSL + supervised + inference + CV)
polymer_comp/              # (optional) place Kaggle data here, or pass --data_dir to a zip/folder
external_dir/              # (optional) extra data like PI1M/Tg/Tc/Density tables
```
You can also point `--data_dir` to a single **zip** that contains `train.csv`, `test.csv`, and `train_supplement/`.

## Environment
- Python 3.10+
- PyTorch 2.x (CUDA preferred), pandas, numpy, scikit-learn, tqdm
- RDKit (for featurization)

A quick install (Linux/Mac with conda):
```bash
conda create -n polygnn python=3.10 -y
conda activate polygnn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick CUDA build that fits your GPU
pip install pandas numpy scikit-learn tqdm rdkit-pypi
```

## Quick start

### 1) Fine‑tune only (single run)
```bash
python train_polymer_gnn_pro.py   --data_dir /path/to/kaggle_data_or_zip   --out_dir ./runs/run1   --finetune_only
```
This will train, save the best checkpoint, and write a `submission_*.csv` to `out_dir`.

### 2) SSL pretraining + supervised
```bash
python train_polymer_gnn_pro.py   --data_dir /path/to/kaggle_data_or_zip   --out_dir ./runs/ssl_run   --pretrain
```
Default SSL uses InfoNCE with node/edge drop augmentations; then fine‑tunes on labeled data.

### 3) Cross‑validation with blending
```bash
python train_polymer_gnn_pro.py   --data_dir /path/to/kaggle_data_or_zip   --out_dir ./runs/cv5   --pretrain   --n_folds 5   --seed_list "11,22,33"
```
The script will train `(folds × seeds)` runs and save a blended `submission.csv` in `out_dir`.

## Notable flags
- `--poly_edge_mode {pair,cycle,clique}`: polymer connectivity variant (default `cycle`).  
- `--epochs`, `--lr`, `--hidden`, `--layers`: supervised training hyperparams.  
- `--hetero_sigma_head`: enable heteroscedastic head (predicts σ).  
- `--physics_reg_lambda`: soft constraint encouraging `Density ≈ 1 − FFV` (set 0 to disable).  
- `--swa`, `--select_model {ema,swa}`: SWA support; by default EMA weights are saved.  
- `--tta N`: test‑time augmentation via MC‑Dropout (N repeats).  
- `--apply_winsor`, `--range_match`: post‑processing from validation stats.  
- `--knn_alpha 0.2 --knn_k 32 --knn_tau 0.2`: optional kNN blend on SSL embeddings.  
- `--cpu`: force CPU run (for debugging).

Run `python train_polymer_gnn_pro.py -h` for the full list.

## Repro tips
- Use the same **SMILES canonicalization** and **scaffold split** for fair validation.  
- For CV: keep `--split_seed` fixed and vary `--seed_list` for model seeds.  
- Cache is stored in `out_dir` (`cache_all_graphs_*.pt`, scaffold maps, kfold indices). You can safely reuse between runs.

## Results
- **Private LB**: Silver medal, **49 / 2240 teams**.  
- Reproducibility: with the defaults plus CV+blending and SSL enabled, you should obtain a strong score close to the silver solution (exact LB varies).

## Acknowledgements
- Competition host: University of Notre Dame & NeurIPS ML4Materials community.
- Libraries: PyTorch, RDKit, scikit‑learn.
- Data: Kaggle competition datasets.

---

*If you use parts of this code for research or competitions, a citation to this repo would be appreciated.*
