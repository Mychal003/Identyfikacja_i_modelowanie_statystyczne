"""
main.py
=======
Główny skrypt projektu nr 11:
  "Identyfikacja nieliniowych systemów dynamicznych
   z wykorzystaniem rekurencyjnych sieci neuronowych"

Uruchomienie:
    python main.py

Wymagania:
    pip install torch torchvision torchaudio scipy matplotlib numpy

CUDA jest używane automatycznie jeśli dostępne.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_generation import build_dataloaders, generate_dataset
from models import build_models
from train import train_model, eval_epoch
from evaluate import (
    plot_learning_curves,
    evaluate_multistep,
    plot_predictions,
    plot_error_horizon,
    print_comparison_table
)

# ─────────────────────────────────────────────
#  Konfiguracja
# ─────────────────────────────────────────────

CONFIG = {
    # Systemy do identyfikacji
    'systems': ['logistic_map'],

    # Hiperparametry sieci
    'hidden_size': 64,
    'num_layers':  2,
    'dropout':     0.1,

    # Hiperparametry treningu
    'seq_len':    50,
    'pred_len':   1,
    'batch_size': 64,
    'n_epochs':   300,
    'lr':         1e-3,
    'patience':   25,

    # Ewaluacja – liczba kroków predykcji per system
    'n_predict': {
        'van_der_pol':  500,
        'duffing':      500,
        'logistic_map': 50,    # chaos eksploduje szybko → krótki horyzont
    },

    # Ścieżki zapisu
    'output_dir': 'wyniki',
}

def main():
    # Utwórz katalog na wyniki
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*60}")
    print(f"  PROJEKT 11: Identyfikacja systemów dynamicznych (RNN)")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'═'*60}")

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    for system_name in CONFIG['systems']:
        print(f"\n{'▓'*60}")
        print(f"  SYSTEM: {system_name.replace('_', ' ').upper()}")
        print(f"{'▓'*60}")

        # ── 1. Dane ──────────────────────────────────────────────
        train_loader, val_loader, train_ds = build_dataloaders(
            system_name=system_name,
            seq_len=CONFIG['seq_len'],
            pred_len=CONFIG['pred_len'],
            batch_size=CONFIG['batch_size'],
        )

        # Dane walidacyjne do predykcji wielokrokowej
        val_trajs, t_val = generate_dataset(
            system_name=system_name,
            n_trajectories=5,
            t_span=(0, 60),
            dt=0.01
        )

        # ── 2. Modele ─────────────────────────────────────────────
        models = build_models(
            input_size=2,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            pred_len=CONFIG['pred_len'],
            dropout=CONFIG['dropout'],
        )

        all_results[system_name] = {}
        histories = {}

        for model_name, model in models.items():
            save_path = os.path.join(
                CONFIG['output_dir'],
                f"best_{system_name}_{model_name.lower()}.pt"
            )

            # ── 3. Trening ────────────────────────────────────────
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=CONFIG['n_epochs'],
                lr=CONFIG['lr'],
                patience=CONFIG['patience'],
                device=device,
                save_path=save_path,
            )
            histories[model_name] = history

            # ── 4. Finalna ewaluacja na walidacji ─────────────────
            _, final_metrics = eval_epoch(
                model, val_loader, nn.MSELoss(), device
            )
            all_results[system_name][model_name] = {
                'history': history,
                'metrics': final_metrics,
                'model':   model,
            }

        # ── 5. Wykresy krzywych uczenia ───────────────────────────
        plot_learning_curves(
            histories=histories,
            system_name=system_name,
            save_path=os.path.join(
                CONFIG['output_dir'],
                f"{system_name}_learning_curves.png"
            )
        )

        # ── 6. Predykcja wielokrokowa ─────────────────────────────
        trained_models = {
            name: data['model']
            for name, data in all_results[system_name].items()
        }
        pred_results, ground_truth = evaluate_multistep(
            models=trained_models,
            dataset=train_ds,
            trajectories_val=val_trajs,
            t_val=t_val,
            seq_len=CONFIG['seq_len'],
            n_predict=min(CONFIG['n_predict'][system_name], len(t_val) - CONFIG['seq_len'] - 1),
            device=device,
        )

        plot_predictions(
            results=pred_results,
            ground_truth=ground_truth,
            t=t_val,
            system_name=system_name,
            seq_len=CONFIG['seq_len'],
            save_path=os.path.join(
                CONFIG['output_dir'],
                f"{system_name}_predictions.png"
            )
        )

        plot_error_horizon(
            results=pred_results,
            ground_truth=ground_truth,
            system_name=system_name,
            save_path=os.path.join(
                CONFIG['output_dir'],
                f"{system_name}_error_horizon.png"
            )
        )

    # ── 7. Tabela porównawcza ─────────────────────────────────────
    print_comparison_table(all_results)

    print(f"\n  ✓  Wszystkie wyniki zapisane w katalogu: '{CONFIG['output_dir']}/'")
    print(f"  Pliki PNG: krzywe uczenia, predykcje, błąd vs horyzont")
    print(f"  Pliki .pt: najlepsze wagi modeli\n")

    plt.show()


if __name__ == '__main__':
    main()