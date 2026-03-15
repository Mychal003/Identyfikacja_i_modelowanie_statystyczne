# Projekt 11 – Identyfikacja nieliniowych systemów dynamicznych z wykorzystaniem rekurencyjnych sieci neuronowych

## Struktura projektu

```
projekt11/
│
├── data_generation.py   # Układy dynamiczne (Van der Pol, Duffing) + Dataset PyTorch
├── models.py            # Modele LSTM i GRU
├── train.py             # Pętla treningowa, Early Stopping, metryki
├── evaluate.py          # Wizualizacje i tabela porównawcza
├── main.py              # Skrypt główny – uruchamia cały eksperyment
│
├── requirements.txt
└── wyniki/              # Generowany katalog z wynikami (PNG, .pt)
```

## Instalacja

```bash
pip install -r requirements.txt
```

Dla CUDA (jeśli masz GPU NVIDIA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Uruchomienie

```bash
python main.py
```

## Badane układy dynamiczne

### 1. Oscylator Van der Pola
```
x'' - μ(1 - x²)x' + x = 0,   μ = 1.5
```
Charakterystyczna cecha: **granica cyklu** – niezależnie od warunków początkowych,
trajektoria dąży do stacjonarnej orbity w przestrzeni fazowej.

### 2. Oscylator Duffinga
```
x'' + δx' + αx + βx³ = γcos(ωt)
δ=0.2, α=-1, β=1, γ=0.3, ω=1.2
```
Charakterystyczna cecha: **chaos deterministyczny** – mała zmiana warunków
początkowych daje dramatycznie różne trajektorie.

## Architektura sieci

Oba modele (LSTM i GRU) mają identyczną pojemność:
- Wejście: `(batch, seq_len=50, 2)` – okno 50 kroków stanu `[x, x']`
- 2 warstwy rekurencyjne, `hidden_size=64`
- Dropout 0.1 między warstwami
- Warstwa liniowa na wyjście: predykcja następnego stanu `[x, x']`
- W fazie testowej: **predykcja autoregresyjna** (sieć dostarcza sobie dane)

## Wyniki (generowane)

| Plik                                | Zawartość                                      |
|-------------------------------------|------------------------------------------------|
| `*_learning_curves.png`             | Krzywe train/val loss (skala log)              |
| `*_predictions.png`                 | x(t), x'(t), portret fazowy vs ODE solver      |
| `*_error_horizon.png`               | RMSE jako funkcja kroku predykcji              |
| Tabela w konsoli                    | MSE / RMSE / MAE dla LSTM i GRU, oba systemy   |

## Wersja rozszerzona (TODO)

- [ ] Ręczna implementacja komórek LSTM i GRU (bez `nn.LSTM`/`nn.GRU`)
- [ ] Wizualizacja bramek LSTM (forget/input/output gate activations)
- [ ] Porównanie z prostszymi baselinesami (vanilla RNN, wielomian)
- [ ] Transfer learning: sieć uczona na Van der Polu → fine-tuning na Duffingu
