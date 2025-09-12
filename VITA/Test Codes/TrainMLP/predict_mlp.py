"""
predict_mlp.py

Objetivo:
Carregar o modelo MLP treinado em train_mlp.py e realizar previsões
em novos dados (ex.: imagens convertidas em AUs) ou classificar imagens diretamente.

Fluxo:
1. Carrega scaler, colunas de AUs e o modelo treinado.
2. Para CSV: Normaliza os dados de entrada no mesmo padrão do treinamento.
3. Para imagem: Extrai AUs usando detector facial e depois classifica.
4. Realiza previsões (autistic / non_autistic).
5. Salva um CSV com os resultados (para CSV) ou retorna predição (para imagem).

Uso:
Para CSV: python predict_mlp.py --csv "./BuildDataset/new_data.csv" --modeldir "TrainMPL/outputs_mlp" --out "predictions.csv"
Para imagem: python predict_mlp.py --image "caminho/para/imagem.jpg" --modeldir "TrainMPL/outputs_mlp"

Dependências:
- O arquivo CSV precisa ter as mesmas colunas de AUs do dataset usado no treino.
- Para classificação de imagem: biblioteca feat (pip install feat)
"""

import argparse
import json
import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
from feat import Detector


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

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Função para classificar imagem
# -----------------------------
def classify_image(image_path, model, scaler, au_columns, label_mapping, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Extrair AUs com o Detector
    detector = Detector()
    preds = detector.detect_image(image_path)

    if preds.empty:
        raise ValueError("Nenhum rosto detectado na imagem!")

    # Extrair apenas as colunas das AUs
    aus = preds.iloc[0].filter(regex="^AU").to_dict()
    
    # Criar DataFrame com as AUs
    X_new = pd.DataFrame([aus])
    
    # Garantir que todas as colunas existam
    for col in au_columns:
        if col not in X_new.columns:
            X_new[col] = 0.0
    
    # Ordenar as colunas
    X_new = X_new[au_columns]
    
    # Normalizar
    X_scaled = scaler.transform(X_new)

    # Converter para tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Predição
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        prediction = preds.cpu().numpy()[0]

    # Mapear label
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    label = reverse_label_mapping.get(prediction, "unknown")

    return {
        "image": image_path,
        "mlp_prediction": label,
        "aus_values": aus
    }


# -----------------------------
# Função para processar CSV
# -----------------------------
def process_csv(csv_path, model, scaler, au_columns, label_mapping, device):
    df = pd.read_csv(csv_path)

    if not set(au_columns).issubset(df.columns):
        raise ValueError("O CSV fornecido não contém todas as colunas de AUs esperadas.")

    # Selecionar colunas corretas
    X_new = df[au_columns].copy()
    X_scaled = scaler.transform(X_new)

    # Converter para tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Predição
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # Mapear labels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    labels_pred = [reverse_label_mapping.get(p, "unknown") for p in preds]

    df["prediction"] = labels_pred
    return df


# -----------------------------
# Função principal
# -----------------------------
def main(args):
    # 1. Carregar artefatos
    scaler_path = os.path.join(args.modeldir, "scaler.pkl")
    label_mapping_path = os.path.join(args.modeldir, "label_mapping.json")
    au_columns_path = os.path.join(args.modeldir, "au_columns.json")
    model_path = os.path.join(args.modeldir, "mlp_model.pt")
    
    if not all(os.path.exists(path) for path in [scaler_path, label_mapping_path, au_columns_path, model_path]):
        raise FileNotFoundError("Arquivos do modelo não encontrados no diretório especificado")
    
    scaler = joblib.load(scaler_path)
    
    with open(label_mapping_path) as f:
        label_mapping = json.load(f)
    
    with open(au_columns_path) as f:
        au_columns = json.load(f)

    # 2. Preparar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=len(au_columns))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 3. Executar conforme argumentos
    if args.csv:
        df = process_csv(args.csv, model, scaler, au_columns, label_mapping, device)
        df.to_csv(args.out, index=False)
        print(f"✅ Previsões salvas em: {args.out}")
        print(df[["prediction"]].head() if "id" not in df.columns else df[["id", "prediction"]].head())

    if args.image:
        result = classify_image(args.image, model, scaler, au_columns, label_mapping, device)
        print("\n🔍 Classificação da imagem:")
        print(f"Imagem: {result['image']}")
        print(f"Predição MLP: {result['mlp_prediction']}")
        print("\nValores das Action Units:")
        for au, val in result['aus_values'].items():
            print(f"  {au}: {val:.4f}")


# -----------------------------
# Argumentos
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Caminho para o CSV com novas imagens convertidas em AUs")
    parser.add_argument("--modeldir", type=str, required=True, help="Diretório contendo os artefatos treinados")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Arquivo de saída para salvar as previsões")
    parser.add_argument("--image", type=str, help="Caminho para uma imagem para classificação")
    
    args = parser.parse_args()
    
    # Verificar se pelo menos uma opção foi fornecida
    if not args.csv and not args.image:
        parser.error("Pelo menos uma das opções --csv ou --image deve ser fornecida")
    
    main(args)
