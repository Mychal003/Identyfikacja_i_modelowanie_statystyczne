"""
data_generation.py
==================
Generowanie danych treningowych z nieliniowych systemów dynamicznych:
  - Oscylator Van der Pola
  - Oscylator Duffinga

Używamy scipy.integrate.solve_ivp jako "prawdziwego" modelu (ground truth).
"""

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Definicje układów dynamicznych
# ─────────────────────────────────────────────

def van_der_pol(t, y, mu=1.0):
    """
    Oscylator Van der Pola:
        x'' - mu*(1 - x^2)*x' + x = 0

    Stan: y = [x, x']
    """
    x, xdot = y
    return [xdot, mu * (1.0 - x**2) * xdot - x]


def duffing(t, y, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2):
    """
    Oscylator Duffinga (wymuszony, tłumiony):
        x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)

    Stan: y = [x, x']
    """
    x, xdot = y
    return [xdot, gamma * np.cos(omega * t) - delta * xdot - alpha * x - beta * x**3]


# ─────────────────────────────────────────────
#  Generacja trajektorii
# ─────────────────────────────────────────────

def generate_trajectory(system_fn, y0, t_span, dt=0.01, **kwargs):
    """
    Całkuje ODE i zwraca równomiernie spróbkowaną trajektorię.

    Parametry
    ---------
    system_fn : callable  – prawa strona ODE f(t, y)
    y0        : list      – warunki początkowe
    t_span    : tuple     – (t_start, t_end)
    dt        : float     – krok czasowy próbkowania
    **kwargs              – parametry systemu (mu, delta, ...)

    Zwraca
    ------
    t  : np.ndarray  shape (N,)
    y  : np.ndarray  shape (N, dim)
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        fun=lambda t, y: system_fn(t, y, **kwargs),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    return sol.t, sol.y.T   # y: (N, dim)


def generate_dataset(system_name='van_der_pol', n_trajectories=20,
                     t_span=(0, 50), dt=0.01,
                     noise_std=0.0):
    """
    Generuje zestaw trajektorii z losowymi warunkami początkowymi.

    Zwraca
    ------
    trajectories : list of np.ndarray, każda shape (N, 2)
    t            : np.ndarray shape (N,)
    """
    rng = np.random.default_rng(42)
    trajectories = []

    for _ in range(n_trajectories):
        if system_name == 'van_der_pol':
            y0 = rng.uniform(-3, 3, size=2).tolist()
            t, y = generate_trajectory(van_der_pol, y0, t_span, dt, mu=1.5)
        elif system_name == 'duffing':
            y0 = rng.uniform(-1, 1, size=2).tolist()
            t, y = generate_trajectory(duffing, y0, t_span, dt,
                                       delta=0.2, alpha=-1.0, beta=1.0,
                                       gamma=0.3, omega=1.2)
        else:
            raise ValueError(f"Nieznany system: {system_name}")

        if noise_std > 0:
            y = y + rng.normal(0, noise_std, size=y.shape)

        trajectories.append(y)

    return trajectories, t


# ─────────────────────────────────────────────
#  PyTorch Dataset – okno przesuwne
# ─────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Zamienia trajektorie na pary (X, y) metodą okna przesuwnego.

    Dla każdej trajektorii:
        X[i] = trajektoria[i : i+seq_len]        shape (seq_len, n_features)
        y[i] = trajektoria[i+seq_len : i+seq_len+pred_len]  shape (pred_len, n_features)
    """

    def __init__(self, trajectories, seq_len=50, pred_len=1, normalize=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples_X = []
        self.samples_y = []

        # Normalizacja globalna (mean/std liczone po wszystkich trajektoriach)
        if normalize:
            all_data = np.concatenate(trajectories, axis=0)
            self.mean = all_data.mean(axis=0)
            self.std  = all_data.std(axis=0) + 1e-8
        else:
            self.mean = np.zeros(trajectories[0].shape[1])
            self.std  = np.ones(trajectories[0].shape[1])

        for traj in trajectories:
            traj_norm = (traj - self.mean) / self.std
            T = len(traj_norm)
            for i in range(T - seq_len - pred_len + 1):
                X = traj_norm[i : i + seq_len]
                y = traj_norm[i + seq_len : i + seq_len + pred_len]
                self.samples_X.append(X)
                self.samples_y.append(y)

        self.samples_X = np.array(self.samples_X, dtype=np.float32)
        self.samples_y = np.array(self.samples_y, dtype=np.float32)

    def __len__(self):
        return len(self.samples_X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples_X[idx]),
            torch.from_numpy(self.samples_y[idx])
        )

    def denormalize(self, y_norm):
        """Odwraca normalizację (tensor lub numpy)."""
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std  = torch.tensor(self.std,  dtype=torch.float32)
        return y_norm * std + mean


def build_dataloaders(system_name, seq_len=50, pred_len=1,
                      batch_size=64, train_split=0.8):
    """
    Buduje DataLoadery treningowy i walidacyjny dla danego systemu.
    """
    print(f"\n[DATA] Generuję trajektorie dla systemu: {system_name.upper()}")
    trajectories, t = generate_dataset(
        system_name=system_name,
        n_trajectories=30,
        t_span=(0, 60),
        dt=0.01,
        noise_std=0.02   # mały szum – bardziej realistyczne dane
    )
    n_train = int(len(trajectories) * train_split)
    train_traj = trajectories[:n_train]
    val_traj   = trajectories[n_train:]

    train_ds = TimeSeriesDataset(train_traj, seq_len=seq_len, pred_len=pred_len)
    val_ds   = TimeSeriesDataset(val_traj,   seq_len=seq_len, pred_len=pred_len,
                                  normalize=False)
    # Val używa tej samej normalizacji co train
    val_ds.mean = train_ds.mean
    val_ds.std  = train_ds.std
    # Przeliczamy val z właściwą normalizacją
    val_ds = TimeSeriesDataset(val_traj, seq_len=seq_len, pred_len=pred_len)
    val_ds.mean = train_ds.mean
    val_ds.std  = train_ds.std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"  Próbki treningowe: {len(train_ds)}")
    print(f"  Próbki walidacyjne: {len(val_ds)}")
    return train_loader, val_loader, train_ds


if __name__ == '__main__':
    # Szybki test – wizualizacja trajektorii
    for name in ['van_der_pol', 'duffing']:
        trajs, t = generate_dataset(name, n_trajectories=3, t_span=(0, 30))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for traj in trajs:
            axes[0].plot(t, traj[:, 0], alpha=0.7)
            axes[1].plot(traj[:, 0], traj[:, 1], alpha=0.7)
        axes[0].set(title=f'{name} – szereg czasowy x(t)',
                    xlabel='t', ylabel='x')
        axes[1].set(title=f'{name} – portret fazowy',
                    xlabel='x', ylabel="x'")
        plt.suptitle(name.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(f'/home/claude/projekt11/{name}_data_preview.png', dpi=120)
        print(f"  Zapisano podgląd danych: {name}_data_preview.png")
    plt.show()
