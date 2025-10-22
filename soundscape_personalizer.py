# soundscape_personalizer.py
# ML Portfolio — Autism-Friendly Soundscape Personalizer
# Reproducible end-to-end: data gen → baselines → LSTM → evaluation → recommendation

import argparse, json, math, os, random, sys
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_curve
)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------- Globals ---------------------
RNG = np.random.default_rng(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIVITIES = ["reading", "maths", "art", "transition", "group_work"]

# --------------------- Utilities ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def moving_average(x, k=5):
    if k <= 1: return x
    pad = np.pad(x, (k//2, k//2), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(pad, kernel, mode="valid")

# --------------------- Synthetic Data ---------------------
def generate_synthetic_day(T=600, base_noise=45.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Time-of-day features
    tod = (t % 1200) / 1200.0
    sin_t = np.sin(2 * np.pi * tod)
    cos_t = np.cos(2 * np.pi * tod)

    # Activity sequence
    activity = rng.choice(ACTIVITIES, size=T, p=[0.26, 0.22, 0.16, 0.18, 0.18])

    # Base noise shaped by activity
    activity_noise = {"reading": -5.0, "maths": 0.0, "art": 1.5, "transition": 6.0, "group_work": 4.0}
    noise = base_noise + np.array([activity_noise[a] for a in activity]) + 2.5 * sin_t + rng.normal(0, 1.6, size=T)
    noise = moving_average(noise, k=7)

    # Background music (change every 20 steps)
    tempo = np.zeros(T); valence = np.zeros(T); arousal = np.zeros(T); centroid = np.zeros(T); lufs = np.zeros(T)
    for i in range(0, T, 20):
        tempo[i:i+20] = rng.integers(60, 130)
        valence[i:i+20] = rng.uniform(0.2, 0.85)
        arousal[i:i+20] = rng.uniform(0.2, 0.9)
        centroid[i:i+20] = rng.uniform(800, 3500)
        lufs[i:i+20] = rng.uniform(-28, -14)   # quieter (more negative) to louder

    # Latent risk logit and binary outcome (next-interval overstimulation)
    activity_risk = {"reading": -0.8, "maths": -0.2, "art": 0.0, "transition": 1.1, "group_work": 0.7}
    recent_noise = moving_average(noise, k=5)
    z = (-2.0
         + 0.06 * (recent_noise - 45)
         + 0.8 * np.array([activity_risk[a] for a in activity])
         + 0.6 * (arousal - 0.5)
         + 0.00025 * (centroid - 2000)
         + 0.001 * (tempo - 90)
         + 0.02 * (lufs + 22)                 # louder (less negative) slightly ↑ risk
         + 0.4 * rng.normal(0, 1, size=T))
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)

    df = pd.DataFrame({
        "noise_db": noise,
        "activity": activity,
        "tempo_bpm": tempo,
        "valence": valence,
        "arousal": arousal,
        "spectral_centroid": centroid,
        "loudness_lufs": lufs,
        "sin_t": sin_t,
        "cos_t": cos_t,
        "target_overstim": y,
    })
    return df

def generate_dataset(days=30, T=600, seed=123):
    rng = np.random.default_rng(seed)
    dfs = []
    for d in range(days):
        base_noise = rng.uniform(42, 48)
        dfs.append(generate_synthetic_day(T=T, base_noise=base_noise, seed=seed + d))
    return pd.concat(dfs, ignore_index=True)

# --------------------- Preprocess ---------------------
@dataclass
class PrepArtifacts:
    ohe: OneHotEncoder
    scaler: StandardScaler
    feature_names: List[str]

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, PrepArtifacts]:
    cat = df[["activity"]]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_arr = ohe.fit_transform(cat)
    cat_cols = [f"act_{n.split('=')[-1]}" for n in ohe.get_feature_names_out(["activity"]).tolist()]

    num_cols = ["noise_db","tempo_bpm","valence","arousal","spectral_centroid","loudness_lufs","sin_t","cos_t"]
    scaler = StandardScaler()
    num_arr = scaler.fit_transform(df[num_cols].values)

    X = np.hstack([num_arr, cat_arr])
    cols = num_cols + cat_cols
    Xdf = pd.DataFrame(X, columns=cols)
    Xdf["y"] = df["target_overstim"].astype(int).values
    return Xdf, PrepArtifacts(ohe=ohe, scaler=scaler, feature_names=cols)

def add_window_aggregates(Xdf: pd.DataFrame, window:int=10) -> pd.DataFrame:
    """Create tabular aggregates over a sliding window for baselines."""
    X = Xdf.copy()
    y = X.pop("y").values
    # compute simple rolling means on numeric columns only
    num_cols = [c for c in X.columns if not c.startswith("act_")]
    roll = X[num_cols].rolling(window=window, min_periods=1).mean()
    roll.columns = [f"{c}_mean{window}" for c in num_cols]
    Xagg = pd.concat([X, roll], axis=1).iloc[window-1:].reset_index(drop=True)
    y = y[window-1:]
    Xagg["y"] = y
    return Xagg

# --------------------- Sequences ---------------------
def make_sequences(Xy: pd.DataFrame, window:int=20, horizon:int=1):
    X = Xy.drop(columns=["y"]).values.astype(np.float32)
    y = Xy["y"].values.astype(np.int64)
    seqX, seqY = [], []
    for i in range(len(X) - window - horizon + 1):
        seqX.append(X[i:i+window])
        seqY.append(y[i+window+horizon-1])
    return np.asarray(seqX, dtype=np.float32), np.asarray(seqY, dtype=np.int64)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# --------------------- Models ---------------------
class LSTMHead(nn.Module):
    def __init__(self, d_in, hidden=64, layers=1, bidir=False, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidir, dropout=0.0 if layers==1 else dropout)
        self.do = nn.Dropout(dropout)
        self.out = nn.Linear(hidden*(2 if bidir else 1), 1)
    def forward(self, x):
        # x: [B, T, D]
        o,_ = self.lstm(x)
        last = o[:, -1, :]
        last = self.do(last)
        logit = self.out(last)
        return logit.squeeze(-1)

# --------------------- Training & Evaluation ---------------------
def evaluate_probs(y_true, y_prob) -> Dict[str,float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
    }

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    bins = np.linspace(0., 1., n_bins+1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0; total = len(y_true)
    for b in range(n_bins):
        idx = inds == b
        if not np.any(idx): continue
        conf = np.mean(y_prob[idx])
        acc = np.mean(y_true[idx])
        ece += (np.sum(idx) / total) * abs(acc - conf)
    return float(ece)

def plot_reliability(y_true, y_prob, outpath):
    bins = np.linspace(0,1,11)
    inds = np.digitize(y_prob, bins)-1
    xs, ys = [], []
    for b in range(10):
        idx = inds==b
        if np.any(idx):
            xs.append(np.mean(y_prob[idx]))
            ys.append(np.mean(y_true[idx]))
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.scatter(xs, ys)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Diagram")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def train_lstm(model, train_loader, val_loader, epochs=8, lr=1e-3, pos_weight=1.0):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    best_pr = -1.0; best_state = None
    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.float().to(DEVICE)
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        # validate
        model.eval(); preds=[]; trues=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                prob = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                preds.append(prob); trues.append(yb.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        pr = average_precision_score(trues, preds)
        roc = roc_auc_score(trues, preds)
        print(f"Epoch {ep}: loss={np.mean(losses):.4f}  valPR={pr:.3f}  valROC={roc:.3f}")
        if pr > best_pr:
            best_pr = pr
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# --------------------- Recommendation ---------------------
def recommend_tracks(model, prep:PrepArtifacts, feature_cols:List[str],
                     context_row:pd.Series, catalogue:pd.DataFrame,
                     goal:str="calm", lam:float=0.3, window:int=20):
    """
    Simple policy: score = predicted_risk + λ * mismatch(goal, track)
    """
    # Define goal template
    # calm: lower arousal, moderate/low tempo, lower centroid, quieter (more negative LUFS)
    # focus: mid arousal, mid tempo, mid centroid
    # energize: higher arousal/tempo/centroid (but this usually increases risk)
    def mismatch(goal, track):
        if goal=="calm":
            target = {"arousal":0.3, "tempo_bpm":75, "spectral_centroid":1200, "loudness_lufs":-24, "valence":0.5}
        elif goal=="focus":
            target = {"arousal":0.45,"tempo_bpm":90,"spectral_centroid":1800,"loudness_lufs":-22,"valence":0.55}
        else: # energize
            target = {"arousal":0.65,"tempo_bpm":105,"spectral_centroid":2400,"loudness_lufs":-20,"valence":0.6}
        diffs = []
        for k,v in target.items():
            diffs.append((float(track[k])-v)**2)
        return math.sqrt(sum(diffs)/len(diffs))

    # Build sequence window from repeated context (toy demo)
    # We replace the last step music features with the candidate track's descriptors.
    numeric = ["noise_db","tempo_bpm","valence","arousal","spectral_centroid","loudness_lufs","sin_t","cos_t"]
    cat = ["activity"]
    seq_rows = []
    for _ in range(window-1):
        seq_rows.append(context_row.copy())
    outs = []
    for _, tr in catalogue.iterrows():
        row = context_row.copy()
        for k in ["tempo_bpm","valence","arousal","spectral_centroid","loudness_lufs"]:
            row[k] = tr[k]
        block = pd.DataFrame(seq_rows + [row])
        # transform
        num_arr = prep.scaler.transform(block[numeric].values)
        cat_arr = prep.ohe.transform(block[cat].values)
        X = np.hstack([num_arr, cat_arr]).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(X[None,...]).to(DEVICE)
            prob = torch.sigmoid(model(inp)).item()
        score = prob + lam * mismatch(goal, tr)
        outs.append((tr["track_id"], prob, score))
    outs.sort(key=lambda x: x[2])
    return outs

# --------------------- Orchestration ---------------------
def run_train_eval(args):
    set_seed(args.seed)
    ensure_dir("results")

    # Data
    df = generate_dataset(days=args.days, T=args.T, seed=args.seed)
    Xdf, prep = prepare_features(df)

    # Train/val/test split by contiguous blocks to avoid leakage
    n = len(Xdf)
    train_end = int(0.6*n); val_end = int(0.8*n)
    X_train = Xdf.iloc[:train_end].reset_index(drop=True)
    X_val   = Xdf.iloc[train_end:val_end].reset_index(drop=True)
    X_test  = Xdf.iloc[val_end:].reset_index(drop=True)

    # ---------- Baselines (tabular aggregates) ----------
    Xagg_train = add_window_aggregates(X_train, window=args.baseline_window)
    Xagg_val   = add_window_aggregates(pd.concat([X_train.tail(args.baseline_window-1), X_val], ignore_index=True), window=args.baseline_window)
    Xagg_test  = add_window_aggregates(pd.concat([X_val.tail(args.baseline_window-1), X_test], ignore_index=True), window=args.baseline_window)

    X_tr = Xagg_train.drop(columns=["y"]).values
    y_tr = Xagg_train["y"].values
    X_va = Xagg_val.drop(columns=["y"]).values
    y_va = Xagg_val["y"].values
    X_te = Xagg_test.drop(columns=["y"]).values
    y_te = Xagg_test["y"].values

    # Logistic Regression
    lr = LogisticRegression(max_iter=200, class_weight="balanced")
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)[:,1]
    lr_metrics = evaluate_probs(y_te, lr_prob)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:,1]
    rf_metrics = evaluate_probs(y_te, rf_prob)

    # ---------- LSTM (sequence) ----------
    # recompute sequences over full (train/val/test) with window args.window
    Xy_train = X_train.copy()
    Xy_val = X_val.copy(); Xy_test = X_test.copy()
    d_in = len(prep.feature_names)

    # Transform to tensors (already standardised in Xdf)
    def to_seq(Xpart):
        X = Xpart.drop(columns=["y"]).values.astype(np.float32)
        y = Xpart["y"].values.astype(np.int64)
        # rebuild sequences on-the-fly from already scaled features
        seqX, seqY = [], []
        w = args.window
        for i in range(len(X) - w):
            seqX.append(X[i:i+w]); seqY.append(y[i+w])
        return np.asarray(seqX), np.asarray(seqY)

    # Note: We use the same standardisation as earlier (already applied), so safe to slice directly
    X_train_scaled = X_train.drop(columns=["y"]).values.astype(np.float32)
    X_val_scaled   = X_val.drop(columns=["y"]).values.astype(np.float32)
    X_test_scaled  = X_test.drop(columns=["y"]).values.astype(np.float32)
    y_train_lab = X_train["y"].values.astype(np.int64)
    y_val_lab   = X_val["y"].values.astype(np.int64)
    y_test_lab  = X_test["y"].values.astype(np.int64)

    def make_seq_from_scaled(Xscaled, ylab, w):
        seqX, seqY = [], []
        for i in range(len(Xscaled)-w):
            seqX.append(Xscaled[i:i+w]); seqY.append(ylab[i+w])
        return np.asarray(seqX), np.asarray(seqY)

    seqX_tr, seqY_tr = make_seq_from_scaled(X_train_scaled, y_train_lab, args.window)
    seqX_va, seqY_va = make_seq_from_scaled(X_val_scaled,   y_val_lab,   args.window)
    seqX_te, seqY_te = make_seq_from_scaled(X_test_scaled,  y_test_lab,  args.window)

    train_ds = SeqDataset(seqX_tr, seqY_tr)
    val_ds   = SeqDataset(seqX_va, seqY_va)
    test_ds  = SeqDataset(seqX_te, seqY_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    pos_weight = max(1.0, (len(seqY_tr)-seqY_tr.sum())/max(1,seqY_tr.sum()))
    lstm = LSTMHead(d_in=d_in, hidden=args.hidden, layers=1, bidir=False, dropout=0.1)
    lstm = train_lstm(lstm, train_loader, val_loader, epochs=args.epochs, lr=args.lr, pos_weight=pos_weight)

    lstm.eval(); lstm_probs=[]; y_true=[]
    with torch.no_grad():
        for xb, yb in test_loader:
            prob = torch.sigmoid(lstm(xb.to(DEVICE))).cpu().numpy()
            lstm_probs.append(prob); y_true.append(yb.numpy())
    lstm_prob = np.concatenate(lstm_probs); y_true = np.concatenate(y_true)
    lstm_metrics = evaluate_probs(y_true, lstm_prob)

    # Calibration
    ece = expected_calibration_error(y_true, lstm_prob, n_bins=10)
    lstm_metrics["ECE"] = ece
    plot_reliability(y_true, lstm_prob, "results/calibration.png")

    # PR curve
    pr, rc, th = precision_recall_curve(y_true, lstm_prob)
    plt.figure()
    plt.plot(rc, pr)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("LSTM Precision-Recall")
    plt.savefig("results/pr_curve.png", bbox_inches="tight"); plt.close()

    # Save metrics
    all_metrics = {
        "LogReg": lr_metrics,
        "RandomForest": rf_metrics,
        "LSTM": lstm_metrics,
        "config": vars(args)
    }
    with open("results/metrics.json","w") as f:
        json.dump(all_metrics, f, indent=2)
    torch.save(lstm.state_dict(), "results/model_lstm.pt")
    with open("results/prep_cols.json","w") as f:
        json.dump({"feature_names": prep.feature_names}, f)

    print("\n=== Test Metrics ===")
    for k,v in all_metrics.items():
        if k in ["config"]: continue
        print(k, v)
    print("Saved: results/metrics.json, results/model_lstm.pt, results/calibration.png, results/pr_curve.png")

    # --------- Simple ablative study (feature groups) ----------
    groups = {
        "no_music": [c for c in prep.feature_names if c not in ["tempo_bpm","valence","arousal","spectral_centroid","loudness_lufs"]],
        "no_noise": [c for c in prep.feature_names if c!="noise_db"],
        "no_context":[c for c in prep.feature_names if not c.startswith("act_") and c not in ["sin_t","cos_t"]] + [], # keep only music+noise
    }
    ab_res = {}
    for name, keep_cols in groups.items():
        keep_idx = [prep.feature_names.index(c) for c in keep_cols]
        # Rebuild sequences with pruned features
        def slice_and_seq(Xscaled, ylab):
            Xs = Xscaled[:, keep_idx]
            seqX, seqY = [], []
            for i in range(len(Xs)-args.window):
                seqX.append(Xs[i:i+args.window]); seqY.append(ylab[i+args.window])
            return np.asarray(seqX), np.asarray(seqY)
        Xtr = X_train_scaled[:, keep_idx]; Xva = X_val_scaled[:, keep_idx]; Xte = X_test_scaled[:, keep_idx]
        sx_tr, sy_tr = slice_and_seq(X_train_scaled, y_train_lab)
        sx_va, sy_va = slice_and_seq(X_val_scaled,   y_val_lab)
        sx_te, sy_te = slice_and_seq(X_test_scaled,  y_test_lab)
        ds_tr = SeqDataset(sx_tr, sy_tr); ds_va=SeqDataset(sx_va, sy_va); ds_te=SeqDataset(sx_te, sy_te)
        tl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
        te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)
        model = LSTMHead(d_in=len(keep_cols), hidden=args.hidden, layers=1, bidir=False, dropout=0.1)
        model = train_lstm(model, tl, vl, epochs=max(4,args.epochs//2), lr=args.lr, pos_weight=pos_weight)
        model.eval(); probs=[]; trues=[]
        with torch.no_grad():
            for xb,yb in te:
                p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                probs.append(p); trues.append(yb.numpy())
        probs = np.concatenate(probs); trues = np.concatenate(trues)
        ab_res[name] = evaluate_probs(trues, probs)
    with open("results/ablations.json","w") as f:
        json.dump(ab_res, f, indent=2)
    print("Saved: results/ablations.json")

def run_export_data(args):
    df = generate_dataset(days=args.days, T=args.T, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Exported synthetic dataset to {args.out}")

def run_recommend(args):
    # Minimal demo using a fresh dataset + trained model if available
    df = generate_dataset(days=5, T=600, seed=999)
    Xdf, prep = prepare_features(df)
    # Load model if exists; else train quick
    feat_names = prep.feature_names
    d_in = len(feat_names)
    lstm = LSTMHead(d_in=d_in, hidden=args.hidden, layers=1, bidir=False, dropout=0.1)
    if os.path.exists("results/model_lstm.pt"):
        lstm.load_state_dict(torch.load("results/model_lstm.pt", map_location=DEVICE))
        lstm.to(DEVICE).eval()
    else:
        print("No trained model found; quick-training on the fly...")
        n=len(Xdf); w=args.window
        X_train = Xdf.iloc[:int(0.8*n)].reset_index(drop=True)
        X_val   = Xdf.iloc[int(0.8*n):].reset_index(drop=True)
        Xtr = X_train.drop(columns=["y"]).values.astype(np.float32)
        ytr = X_train["y"].values.astype(np.int64)
        Xva = X_val.drop(columns=["y"]).values.astype(np.float32)
        yva = X_val["y"].values.astype(np.int64)
        def mk(X,y,w):
            seqX, seqY = [], []
            for i in range(len(X)-w):
                seqX.append(X[i:i+w]); seqY.append(y[i+w])
            return np.asarray(seqX), np.asarray(seqY)
        sx_tr, sy_tr = mk(Xtr, ytr, w); sx_va, sy_va = mk(Xva, yva, w)
        tl = DataLoader(SeqDataset(sx_tr, sy_tr), batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(SeqDataset(sx_va, sy_va), batch_size=args.batch_size, shuffle=False)
        pos_weight = max(1.0, (len(sy_tr)-sy_tr.sum())/max(1,sy_tr.sum()))
        lstm = train_lstm(lstm, tl, vl, epochs=6, lr=args.lr, pos_weight=pos_weight)
        torch.save(lstm.state_dict(), "results/model_lstm.pt")
        lstm.eval()

    # Build a tiny track catalogue
    cat = pd.DataFrame([
        {"track_id":"calm_piano","tempo_bpm":72,"valence":0.45,"arousal":0.30,"spectral_centroid":1100,"loudness_lufs":-25},
        {"track_id":"soft_strings","tempo_bpm":78,"valence":0.50,"arousal":0.35,"spectral_centroid":1300,"loudness_lufs":-24},
        {"track_id":"lofi_focus","tempo_bpm":88,"valence":0.55,"arousal":0.45,"spectral_centroid":1700,"loudness_lufs":-22},
        {"track_id":"bright_uplift","tempo_bpm":108,"valence":0.65,"arousal":0.70,"spectral_centroid":2600,"loudness_lufs":-20},
    ])

    # Choose the latest context row (e.g., now)
    last = df.iloc[-1].copy()
    # Make sure activity exists
    context = last.copy()

    recs = recommend_tracks(
        model=lstm, prep=prep, feature_cols=feat_names,
        context_row=context, catalogue=cat,
        goal=args.goal, lam=args.lam, window=args.window
    )
    print("\nRecommended tracks (best first):")
    for tid, risk, score in recs:
        print(f"{tid:16s}  predicted_risk={risk:.3f}  total_score={score:.3f}")

# --------------------- CLI ---------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train_eval", help="Train baselines + LSTM, evaluate, save metrics/plots")
    tr.add_argument("--days", type=int, default=30)
    tr.add_argument("--T", type=int, default=600)
    tr.add_argument("--window", type=int, default=20)
    tr.add_argument("--baseline_window", type=int, default=10)
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=42)

    ex = sub.add_parser("export_data", help="Export synthetic dataset to CSV")
    ex.add_argument("--days", type=int, default=30)
    ex.add_argument("--T", type=int, default=600)
    ex.add_argument("--seed", type=int, default=42)
    ex.add_argument("--out", type=str, default="data.csv")

    rc = sub.add_parser("recommend", help="Run recommendation demo")
    rc.add_argument("--window", type=int, default=20)
    rc.add_argument("--goal", type=str, default="calm", choices=["calm","focus","energize"])
    rc.add_argument("--lam", type=float, default=0.3)
    rc.add_argument("--batch_size", type=int, default=64)
    rc.add_argument("--hidden", type=int, default=64)
    rc.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()
    if args.mode == "train_eval":
        run_train_eval(args)
    elif args.mode == "export_data":
        run_export_data(args)
    elif args.mode == "recommend":
        run_recommend(args)
    else:
        print("Unknown mode")

if __name__ == "__main__":
    main()
