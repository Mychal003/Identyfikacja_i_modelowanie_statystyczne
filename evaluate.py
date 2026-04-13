"""
evaluate.py
===========
Ewaluacja wytrenowanych modeli i wizualizacja wyników:
  1. Krzywe uczenia (train/val loss)
  2. Porównanie predykcji z ground truth (solver ODE)
  3. Portret fazowy – predykcja vs rzeczywistość
  4. Tabela porównawcza LSTM vs GRU
  5. Błąd predykcji jako funkcja horyzontu czasowego
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


# ─────────────────────────────────────────────
#  Konfiguracja stylu wykresów
# ─────────────────────────────────────────────

COLORS = {
    'ground_truth': '#2d2d2d',
    'LSTM': '#e63946',
    'GRU':  '#457b9d',
    'train': '#2a9d8f',
    'val':   '#e9c46a',
}

plt.rcParams.update({
    'figure.facecolor': '#fafafa',
    'axes.facecolor':   '#f0f0f0',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})


# ─────────────────────────────────────────────
#  1. Krzywe uczenia
# ─────────────────────────────────────────────

def plot_learning_curves(histories, system_name, save_path=None):
    """
    histories : dict {'LSTM': history_lstm, 'GRU': history_gru}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Krzywe uczenia – {system_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')

    for i, (model_name, history) in enumerate(histories.items()):
        ax = axes[i]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'],
                color=COLORS['train'], label='Train MSE', linewidth=1.5)
        ax.plot(epochs, history['val_loss'],
                color=COLORS['val'], label='Val MSE', linewidth=1.5)
        ax.set(title=f'{model_name}', xlabel='Epoka', ylabel='MSE Loss')
        ax.legend()
        ax.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Zapisano: {save_path}")
    return fig


# ─────────────────────────────────────────────
#  2. Predykcja wielokrokowa vs ground truth
# ─────────────────────────────────────────────

def evaluate_multistep(models, dataset, trajectories_val, t_val,
                        seq_len=50, n_predict=300, device=None):
    """
    Dla każdego modelu: bierze okno startowe z walidacji,
    a następnie autoregresyjnie przewiduje n_predict kroków.

    Zwraca
    ------
    results : dict {'LSTM': pred_array, 'GRU': pred_array}
               każda pred_array shape (n_predict, 2)
    ground_truth : np.ndarray shape (n_predict, 2)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Wybierz trajektorię startową (ostatnią z walidacji)
    traj = trajectories_val[-1]
    traj_norm = (traj - dataset.mean) / dataset.std

    # Okno startowe
    x_init = torch.tensor(
        traj_norm[:seq_len], dtype=torch.float32
    ).unsqueeze(0)   # (1, seq_len, 2)

    ground_truth = traj[seq_len: seq_len + n_predict]   # w oryginalnej skali

    results = {}
    for model_name, model in models.items():
        model.eval()
        pred_norm = model.predict_sequence(x_init, n_predict, device)
        # Odwróć normalizację
        mean = torch.tensor(dataset.mean, dtype=torch.float32)
        std  = torch.tensor(dataset.std,  dtype=torch.float32)
        pred = (pred_norm.cpu() * std + mean).numpy()
        results[model_name] = pred

    return results, ground_truth


def plot_predictions(results, ground_truth, t, system_name,
                     seq_len=50, save_path=None):
    """
    Rysuje: szeregi czasowe x(t) i x'(t) + portret fazowy.
    """
    t_pred = t[seq_len: seq_len + len(ground_truth)]
    n = len(t_pred)

    # Etykiety zależne od systemu
    if system_name == 'logistic_map':
        gt_label   = 'Mapa logistyczna (ground truth)'
        xlabel_ts  = 'n (krok)'
        ylabel_x   = 'xₙ'
        ylabel_xd  = 'xₙ₊₁'
        title_x    = 'xₙ vs krok'
        title_xd   = 'xₙ₊₁ vs krok'
        title_ph   = 'Atraktor (xₙ vs xₙ₊₁)'
        xlabel_ph  = 'xₙ'
        ylabel_ph  = 'xₙ₊₁'
    else:
        gt_label   = 'ODE solver (ground truth)'
        xlabel_ts  = 't'
        ylabel_x   = 'x(t)'
        ylabel_xd  = "x'(t)"
        title_x    = 'Pozycja x(t)'
        title_xd   = "Prędkość x'(t)"
        title_ph   = 'Portret fazowy'
        xlabel_ph  = 'x'
        ylabel_ph  = "x'"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Predykcja wielokrokowa – {system_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_x     = fig.add_subplot(gs[0, :2])
    ax_xdot  = fig.add_subplot(gs[1, :2])
    ax_phase = fig.add_subplot(gs[:, 2])

    # --- Szereg x ---
    ax_x.plot(t_pred[:n], ground_truth[:, 0],
              color=COLORS['ground_truth'], lw=2, label=gt_label, zorder=3)
    for model_name, pred in results.items():
        ax_x.plot(t_pred[:n], pred[:n, 0],
                  color=COLORS[model_name], lw=1.5,
                  linestyle='--', label=model_name, alpha=0.85)
    ax_x.set(xlabel=xlabel_ts, ylabel=ylabel_x, title=title_x)
    ax_x.legend(loc='upper right')

    # --- Szereg x' / x_{n+1} ---
    ax_xdot.plot(t_pred[:n], ground_truth[:, 1],
                 color=COLORS['ground_truth'], lw=2, label=gt_label, zorder=3)
    for model_name, pred in results.items():
        ax_xdot.plot(t_pred[:n], pred[:n, 1],
                     color=COLORS[model_name], lw=1.5,
                     linestyle='--', label=model_name, alpha=0.85)
    ax_xdot.set(xlabel=xlabel_ts, ylabel=ylabel_xd, title=title_xd)
    ax_xdot.legend(loc='upper right')

    # --- Portret fazowy / atraktor ---
    ax_phase.plot(ground_truth[:, 0], ground_truth[:, 1],
                  color=COLORS['ground_truth'], lw=2, label=gt_label, zorder=3)
    for model_name, pred in results.items():
        ax_phase.plot(pred[:n, 0], pred[:n, 1],
                      color=COLORS[model_name], lw=1.5,
                      linestyle='--', label=model_name, alpha=0.85)
    ax_phase.set(xlabel=xlabel_ph, ylabel=ylabel_ph, title=title_ph)
    ax_phase.legend()
    ax_phase.set_aspect('auto')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Zapisano: {save_path}")
    return fig


# ─────────────────────────────────────────────
#  3. Błąd jako funkcja horyzontu
# ─────────────────────────────────────────────

def plot_error_horizon(results, ground_truth, system_name, save_path=None):
    """
    Kumulatywny RMSE jako funkcja kroku predykcji.
    Pokazuje jak szybko błąd narasta w czasie.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f'Błąd predykcji vs horyzont – {system_name.replace("_", " ").title()}',
        fontsize=13, fontweight='bold'
    )
    n = len(ground_truth)
    steps = np.arange(1, n + 1)

    for model_name, pred in results.items():
        rmse_per_step = np.sqrt(
            np.mean((pred[:n] - ground_truth[:n])**2, axis=1)
        )
        ax.plot(steps, rmse_per_step,
                color=COLORS[model_name], lw=2, label=model_name)

    ax.set(xlabel='Krok predykcji', ylabel='RMSE', title='Narastanie błędu w czasie')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Zapisano: {save_path}")
    return fig


# ─────────────────────────────────────────────
#  4. Tabela porównawcza
# ─────────────────────────────────────────────

def print_comparison_table(all_results):
    """
    all_results : dict {system_name: {model_name: {'metrics': ..., 'history': ...}}}
    """
    print(f"\n{'═'*70}")
    print(f"  TABELA PORÓWNAWCZA – LSTM vs GRU")
    print(f"{'═'*70}")
    print(f"  {'System':<20} {'Model':<8} {'Val MSE':>10} "
          f"{'Val RMSE':>10} {'Val MAE':>10} {'Epoki':>7}")
    print(f"{'─'*70}")

    for system_name, model_results in all_results.items():
        for model_name, data in model_results.items():
            metrics = data['metrics']
            n_epochs = len(data['history']['train_loss'])
            print(f"  {system_name:<20} {model_name:<8} "
                  f"{metrics['mse']:>10.6f} "
                  f"{metrics['rmse']:>10.6f} "
                  f"{metrics['mae']:>10.6f} "
                  f"{n_epochs:>7}")
    print(f"{'═'*70}")
