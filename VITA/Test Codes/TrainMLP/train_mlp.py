import argparse
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Definição da rede MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return self.log_softmax(x)

# -----------------------------
# Inicialização de pesos
# -----------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# -----------------------------
# Dataset customizado
# -----------------------------
class FausDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Função de treino e validação
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping para prevenir explosão
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc = float("nan")

    return avg_loss, acc, prec, rec, f1, roc

# -----------------------------
# Função principal
# -----------------------------
def main(args):
    start_time = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Carregar dataset e remover valores NaN
    df = pd.read_csv(args.csv)
    initial_shape = df.shape
    print(f"Dataset original: {initial_shape[0]} linhas, {initial_shape[1]} colunas")
    
    # Remover linhas com valores NaN
    df_clean = df.dropna()
    removed_rows = initial_shape[0] - df_clean.shape[0]
    
    print(f"Removidas {removed_rows} linhas com valores NaN")
    print(f"Dataset limpo: {df_clean.shape[0]} linhas ({removed_rows/initial_shape[0]*100:.1f}% de remoção)")
    
    if removed_rows > 0:
        # Salvar informações sobre dados removidos
        nan_rows = df[df.isnull().any(axis=1)]
        nan_rows[['id', 'label']].to_csv(os.path.join(args.outdir, "removed_nan_rows.csv"), index=False)
        print(f"Informações das linhas removidas salvas em: {os.path.join(args.outdir, 'removed_nan_rows.csv')}")

    # 2. Verificar se ainda temos dados suficientes
    if len(df_clean) < 100:
        print("ERRO: Poucos dados após limpeza! Necessário pelo menos 100 amostras.")
        return

    # 3. Separar features e labels dos dados limpos
    X = df_clean.drop(columns=["id", "label"])
    y = df_clean["label"]

    # 4. Remover colunas com desvio padrão zero
    zero_std_cols = X.columns[X.std() == 0].tolist()
    if zero_std_cols:
        print(f"[AVISO] Colunas com desvio padrão zero removidas: {zero_std_cols}")
        X = X.drop(columns=zero_std_cols)

    # 5. Normalizar features com StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Valores após scaling: min={X_scaled.min():.4f}, max={X_scaled.max():.4f}")
    print(f"NaN após scaling: {np.isnan(X_scaled).sum()}")
    print(f"Inf após scaling: {np.isinf(X_scaled).sum()}")

    # 6. Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Verificar distribuição das classes
    unique, counts = np.unique(y_encoded, return_counts=True)
    print(f"Distribuição das classes: {dict(zip(unique, counts))}")

    # 7. Dividir dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 8. Criar datasets e dataloaders
    train_loader = DataLoader(FausDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(FausDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(FausDataset(X_test, y_test), batch_size=32)

    # 9. Definir modelo, loss e optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1]).to(device)
    model.apply(init_weights)  # Inicialização de pesos
    
    criterion = nn.NLLLoss()  # Use NLLLoss com LogSoftmax
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 10. Treinamento
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_roc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Época {epoch:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.outdir, "mlp_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping ativado na época {epoch}.")
                break

    # 11. Avaliar no teste (se o modelo foi salvo)
    if os.path.exists(os.path.join(args.outdir, "mlp_model.pt")):
        model.load_state_dict(torch.load(os.path.join(args.outdir, "mlp_model.pt")))
        test_loss, test_acc, test_prec, test_rec, test_f1, test_roc = evaluate(model, test_loader, criterion, device)
        metrics = {
            "loss": test_loss,
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
            "roc_auc": test_roc
        }
        print("Métricas no TESTE:", json.dumps(metrics, indent=2))

        # Salvar artefatos
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        print("Modelo não foi salvo devido a problemas no treinamento")

    # Salvar outros artefatos
    joblib.dump(scaler, os.path.join(args.outdir, "scaler.pkl"))
    with open(os.path.join(args.outdir, "label_mapping.json"), "w") as f:
        json.dump({int(i): l for i, l in enumerate(label_encoder.classes_)}, f, indent=2)
    with open(os.path.join(args.outdir, "au_columns.json"), "w") as f:
        json.dump(list(X.columns), f, indent=2)

    # Plotar curva
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Curva de Treinamento - MLP")
    plt.savefig(os.path.join(args.outdir, "training_curve.png"))
    plt.close()

    print(f"Treinamento finalizado em {time.time() - start_time:.1f}s. Artefatos salvos em {args.outdir}")

# -----------------------------
# Argumentos
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Caminho para o dataset CSV")
    parser.add_argument("--outdir", type=str, required=True, help="Diretório para salvar artefatos")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas de treino")
    args = parser.parse_args()
    main(args)