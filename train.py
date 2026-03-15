"""
train.py
========
Pętla treningowa z:
  - Wczesnym zatrzymaniem (Early Stopping)
  - Harmonogramem LR (ReduceLROnPlateau)
  - Zapisem najlepszego modelu
  - Metrykami: MSE, MAE, RMSE
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ─────────────────────────────────────────────
#  Metryki
# ─────────────────────────────────────────────

def compute_metrics(pred, target):
    """
    Oblicza MSE, RMSE, MAE dla dwóch tensorów.

    Parametry
    ---------
    pred, target : Tensor shape (N, ...)

    Zwraca
    ------
    dict z kluczami 'mse', 'rmse', 'mae'
    """
    mse  = nn.functional.mse_loss(pred, target).item()
    mae  = nn.functional.l1_loss(pred, target).item()
    return {'mse': mse, 'rmse': np.sqrt(mse), 'mae': mae}


# ─────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Zatrzymuje trening gdy val_loss przestaje spadać przez 'patience' epok.
    Zapisuje stan najlepszego modelu.
    """

    def __init__(self, patience=15, min_delta=1e-5, path='best_model.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0
        self.best_loss  = np.inf
        self.stop       = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path, map_location='cpu'))
        return model


# ─────────────────────────────────────────────
#  Jedna epoka treningowa
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred, _ = model(X)
        loss = criterion(pred, y)
        loss.backward()
        # Gradient clipping – zapobiega eksplozji gradientów w RNN
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item() * X.size(0)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return total_loss / len(loader.dataset), compute_metrics(preds, targets)


# ─────────────────────────────────────────────
#  Główna pętla treningowa
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                n_epochs=200, lr=1e-3, weight_decay=1e-4,
                patience=20, device=None, save_path='best_model.pt'):
    """
    Trenuje model i zwraca historię metryk.

    Zwraca
    ------
    history : dict z kluczami 'train_loss', 'val_loss', 'val_metrics'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=10)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, path=save_path)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    t0 = time.time()

    print(f"\n{'─'*60}")
    print(f"  Model: {model.rnn_type}  |  Device: {device}")
    print(f"  Parametry: {model.count_parameters():,}  |  Epoki: {n_epochs}")
    print(f"{'─'*60}")
    print(f"  {'Epoka':>5}  {'Train MSE':>10}  {'Val MSE':>10}  "
          f"{'Val RMSE':>10}  {'LR':>8}")
    print(f"{'─'*60}")

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  {epoch:>5}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{val_metrics['rmse']:>10.6f}  {current_lr:>8.2e}  "
                  f"[{elapsed:.1f}s]")

        if early_stopping.stop:
            print(f"\n  ⏹  Early stopping po epoce {epoch}")
            break

    # Załaduj najlepszy model
    model = early_stopping.load_best(model)
    print(f"\n  ✓  Najlepsza val MSE: {early_stopping.best_loss:.6f}")
    print(f"  Czas treningu: {time.time() - t0:.1f}s")
    return history