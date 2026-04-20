"""
main.py
=======
Eksperyment bifurkacyjny – mapa logistyczna dla różnych wartości r.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_generation import generate_logistic_dataset, TimeSeriesDataset
from torch.utils.data import DataLoader
from models import build_models
from train import train_model
from evaluate import evaluate_multistep, plot_predictions, plot_error_horizon

CONFIG = {
    'r_values':   [3.5, 3.7, 3.8, 3.9, 4.0],
    'hidden_size': 64,
    'num_layers':  2,
    'dropout':     0.1,
    'seq_len':     50,
    'n_epochs':    300,
    'lr':          1e-3,
    'patience':    25,
    'n_predict':   200,
    'output_dir':  'wyniki',
}


def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n{'═'*60}")
    print(f"  EKSPERYMENT: Mapa logistyczna – RMSE vs r")
    print(f"  Device: {device}")
    print(f"{'═'*60}")

    results_per_r = {r: {} for r in CONFIG['r_values']}

    for r in CONFIG['r_values']:
        print(f"\n{'─'*50}")
        print(f"  r = {r}")
        print(f"{'─'*50}")

        # ── Dane ──────────────────────────────────────────────
        trajectories, t = generate_logistic_dataset(
            n_trajectories=30, r=r, n_steps=6000, noise_std=0.005
        )
        n_train    = int(len(trajectories) * 0.8)
        train_traj = trajectories[:n_train]
        val_traj   = trajectories[n_train:]

        train_ds = TimeSeriesDataset(train_traj, seq_len=CONFIG['seq_len'], pred_len=1)
        val_ds   = TimeSeriesDataset(val_traj,   seq_len=CONFIG['seq_len'], pred_len=1)
        val_ds.mean, val_ds.std = train_ds.mean, train_ds.std

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

        # ── Modele + trening ───────────────────────────────────
        models = build_models(
            input_size=2,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            pred_len=1,
            dropout=CONFIG['dropout'],
        )

        for model_name, model in models.items():
            save_path = os.path.join(
                CONFIG['output_dir'],
                f"bifurc_r{r}_{model_name.lower()}.pt"
            )
            train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                n_epochs=CONFIG['n_epochs'], lr=CONFIG['lr'],
                patience=CONFIG['patience'], device=device, save_path=save_path
            )

        # ── Predykcja wszystkich modeli naraz ──────────────────
        all_preds, ground_truth = evaluate_multistep(
            models=models,
            dataset=train_ds,
            trajectories_val=val_traj,
            t_val=t,
            seq_len=CONFIG['seq_len'],
            n_predict=CONFIG['n_predict'],
            device=device,
        )

        # ── RMSE per model ─────────────────────────────────────
        n = min(len(next(iter(all_preds.values()))), len(ground_truth))
        for model_name, pred in all_preds.items():
            rmse = float(np.sqrt(np.mean((pred[:n] - ground_truth[:n])**2)))
            results_per_r[r][model_name] = rmse
            print(f"    {model_name}  RMSE = {rmse:.4f}")

        # ── Wykresy dla danego r ───────────────────────────────
        plot_predictions(
            results=all_preds,
            ground_truth=ground_truth,
            t=t,
            system_name='logistic_map',
            seq_len=CONFIG['seq_len'],
            save_path=os.path.join(
                CONFIG['output_dir'],
                f"logistic_r{r}_predictions.png"
            )
        )

        plot_error_horizon(
            results=all_preds,
            ground_truth=ground_truth,
            system_name='logistic_map',
            save_path=os.path.join(
                CONFIG['output_dir'],
                f"logistic_r{r}_error_horizon.png"
            )
        )

    # ── Wykres RMSE vs r ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Mapa logistyczna – RMSE vs parametr r',
                 fontsize=13, fontweight='bold')

    for model_name, color in [('LSTM', '#e63946'), ('GRU', '#457b9d')]:
        rmses = [results_per_r[r][model_name] for r in CONFIG['r_values']]
        ax.plot(CONFIG['r_values'], rmses, marker='o', color=color,
                lw=2, label=model_name)

    ax.set(xlabel='r', ylabel=f"RMSE ({CONFIG['n_predict']} kroków)",
           xticks=CONFIG['r_values'])
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(CONFIG['output_dir'], 'logistic_bifurcation_rmse.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Zapisano: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()