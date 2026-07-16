import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Vytvoření matice pozičních enkodérů
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # Uloží 'pe' do stavu modulu, ale ne jako parametr k trénování
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x má tvar (seq_len, batch_size, d_model)
        # NEBO (batch_size, seq_len, d_model) pokud je batch_first=True

        # Přidáme poziční enkodér k vstupním datům
        # Předpokládáme, že x[0] je dimenze sekvence, pokud batch_first=False
        # Nebo x[1] je dimenze sekvence, pokud batch_first=True
        # Pro náš případ s batch_first=True: x je (batch, seq, embed)
        # pe je (max_len, 1, embed) -> pe[:x.size(1), :] je (seq, 1, embed)

        # Musíme přizpůsobit PE matici pro batch_first
        # self.pe je (max_len, 1, d_model), chceme (1, seq_len, d_model)
        pe_transposed = self.pe.transpose(0, 1)  # (1, max_len, d_model)
        x = x + pe_transposed[:, :x.size(1), :]

        return self.dropout(x)


# --- ČÁST 2: Samotný Transformer Model ---
class HysteresisTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bidirectional, n_heads):
        super(HysteresisTransformer, self).__init__()

        # Stejné proměnné jako v GRU/LSTM
        self.hidden_size = hidden_size  # Toto bude náš 'd_model'
        self.num_layers = num_layers

        # Počet "hlav" v multi-head attention.
        # Musí platit: hidden_size % n_heads == 0
        self.n_heads = n_heads
        assert hidden_size % n_heads == 0, "hidden_size musí být dělitelné počtem hlav (n_heads)."
        # Dimenze vnitřní feed-forward vrstvy
        self.dim_feedforward = hidden_size * 4  # Standardní praxe (např. 128*4 = 512)
        self.dropout = dropout

        # 1. Vstupní Embedding
        # Převede tvůj vstup (input_size) na vnitřní dimenzi modelu (hidden_size)
        self.input_embed = nn.Linear(input_size, hidden_size)

        # 2. Poziční Enkodér
        self.pos_encoder = PositionalEncoding(hidden_size, self.dropout)

        # 3. Samotný Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True  # DŮLEŽITÉ! Pracujeme s (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. Výstupní FC vrstva
        # Na konci zprůměrujeme výstupy ze sekvence a pošleme je do FC vrstvy
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        # Inicializace vah (dobrá praxe)
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src má tvar (batch_size, seq_len, input_size)

        # 1. Projdi vstup embeddingem
        # (batch, seq, input_size) -> (batch, seq, hidden_size)
        src = self.input_embed(src) * math.sqrt(self.hidden_size)  # Škálování je standard

        # 2. Přidej poziční informaci
        src = self.pos_encoder(src)

        # 3. Prožeň data Transformerem
        # (batch, seq, hidden_size) -> (batch, seq, hidden_size)
        output = self.transformer_encoder(src)

        # 4. Agregace
        # Na rozdíl od GRU, kde bereme poslední výstup out[:, -1, :],
        # u Transformeru je robustnější vzít PRŮMĚR všech výstupů v sekvenci.
        # (batch, seq, hidden_size) -> (batch, hidden_size)
        output = output.mean(dim=1)

        # 5. Finální predikce
        # (batch, hidden_size) -> (batch, output_size)
        output = self.fc(output)

        return output