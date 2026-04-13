"""
models.py
=========
Definicje modeli sieci rekurencyjnych (wersja podstawowa – gotowe warstwy PyTorch):
  - RNNPredictor  – ogólna klasa z wymiennym typem komórki (LSTM / GRU)
  - LSTMPredictor – alias z LSTM
  - GRUPredictor  – alias z GRU
"""

import torch
import torch.nn as nn


class RNNPredictor(nn.Module):
    """
    Ogólny prediktor szeregów czasowych oparty na sieci rekurencyjnej.

    Architektura:
        Input  →  [RNN layers]  →  [Dropout]  →  [Linear]  →  Output

    Parametry
    ---------
    input_size  : int   – liczba cech wejściowych (tu: 2, czyli x i x')
    hidden_size : int   – rozmiar stanu ukrytego
    num_layers  : int   – liczba warstw rekurencyjnych
    output_size : int   – rozmiar wyjścia (tu: 2)
    pred_len    : int   – horyzont predykcji (liczba kroków naprzód)
    rnn_type    : str   – 'LSTM' lub 'GRU'
    dropout     : float – prawdopodobieństwo dropout (stosowany między warstwami)
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2,
                 output_size=2, pred_len=1, rnn_type='LSTM', dropout=0.1):
        super().__init__()
        self.rnn_type   = rnn_type.upper()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.pred_len    = pred_len

        rnn_class = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # wejście: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # Wyjście: pred_len kroków × output_size cech
        self.fc = nn.Linear(hidden_size, output_size * pred_len)
        self.output_size = output_size

        self._init_weights()

    def _init_weights(self):
        """Inicjalizacja wag – ortogonalna dla RNN, Xavier dla FC."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Dla LSTM: ustaw forget gate bias = 1 (stabilizuje trening)
                if self.rnn_type == 'LSTM':
                    n = param.size(0)
                    param.data[n // 4: n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        """
        Parametry
        ---------
        x      : Tensor shape (batch, seq_len, input_size)
        hidden : opcjonalny stan ukryty

        Zwraca
        ------
        out    : Tensor shape (batch, pred_len, output_size)
        hidden : stan ukryty po przetworzeniu sekwencji
        """
        rnn_out, hidden = self.rnn(x, hidden)
        # Bierzemy tylko ostatni krok czasowy
        last_hidden = self.dropout(rnn_out[:, -1, :])   # (batch, hidden_size)
        out = self.fc(last_hidden)                        # (batch, pred_len*output_size)
        out = out.view(-1, self.pred_len, self.output_size)
        return out, hidden

    def predict_sequence(self, x_init, n_steps, device):
        """
        Predykcja wielokrokowa (autoregresja): sieć sama dostarcza sobie dane.

        Parametry
        ---------
        x_init  : Tensor shape (1, seq_len, input_size) – okno startowe
        n_steps : int  – ile kroków do przodu przewidzieć
        device  : torch.device

        Zwraca
        ------
        predictions : Tensor shape (n_steps, input_size)
        """
        self.eval()
        predictions = []
        x = x_init.to(device)

        with torch.no_grad():
            for _ in range(n_steps):
                out, _ = self.forward(x, None)       # hidden zawsze None — jak w treningu
                pred = out[:, 0:1, :]                # (1, 1, output_size)
                predictions.append(pred.squeeze(0))  # (1, output_size)
                # Przesuń okno – usuń pierwszy krok, dodaj predykcję
                x = torch.cat([x[:, 1:, :], pred], dim=1)

        return torch.cat(predictions, dim=0)         # (n_steps, output_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"RNNPredictor(type={self.rnn_type}, "
                f"hidden={self.hidden_size}, layers={self.num_layers}, "
                f"params={self.count_parameters():,})")


def build_models(input_size=2, hidden_size=64, num_layers=2,
                 pred_len=1, dropout=0.1):
    """
    Buduje parę modeli LSTM + GRU o identycznej pojemności.

    Zwraca
    ------
    dict: {'LSTM': model_lstm, 'GRU': model_gru}
    """
    models = {}
    for rnn_type in ['LSTM', 'GRU']:
        models[rnn_type] = RNNPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            pred_len=pred_len,
            rnn_type=rnn_type,
            dropout=dropout
        )
    return models


if __name__ == '__main__':
    # Szybki test kształtów tensorów
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    models = build_models()
    for name, model in models.items():
        model = model.to(device)
        print(model)
        x = torch.randn(32, 50, 2).to(device)   # batch=32, seq=50, features=2
        out, _ = model(x)
        print(f"  Wejście: {tuple(x.shape)}  →  Wyjście: {tuple(out.shape)}\n")
