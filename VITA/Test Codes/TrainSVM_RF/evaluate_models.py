"""
Avaliação e predição com modelos treinados (SVM e Random Forest).
Pode ser usado como módulo ou rodado via CLI.
"""

import argparse
import os
import json
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from feat import Detector


# ---------------------------------------------------
# Carregar descrições de AUs
# ---------------------------------------------------
with open("Aus/au_descriptions.json", "r", encoding="utf-8") as f:
    au_descriptions = json.load(f)


# ---------------------------------------------------
# Avaliar no dataset CSV
# ---------------------------------------------------
def evaluate_models(csv_path="BuildDataset/dataset_faus.csv"):
    """
    Avalia os modelos salvos em um dataset CSV.

    Args:
        csv_path (str): Caminho para o dataset CSV.

    Retorna:
        dict com relatórios de classificação.
    """
    df = pd.read_csv(csv_path)
    
    # CORREÇÃO: Usar "id" em vez de "image"
    X = df.drop(columns=["id", "label"])
    y = df["label"].map({"autistic": 1, "non_autistic": 0})

    # Carregar modelos e pré-processadores
    svm_model = joblib.load("TrainSMV_RF/svm_model.pkl")
    rf_model = joblib.load("TrainSMV_RF/rf_model.pkl")
    scaler = joblib.load("TrainSMV_RF/scaler.pkl")
    imputer = joblib.load("TrainSMV_RF/imputer.pkl")  # Carregar o imputer também

    # Aplicar o mesmo pré-processamento do treinamento
    X_imputed = imputer.transform(X)  # Tratar valores NaN
    X_scaled = scaler.transform(X_imputed)  # Normalizar

    y_pred_svm = svm_model.predict(X_scaled)
    y_pred_rf = rf_model.predict(X_scaled)

    svm_report = classification_report(
        y, y_pred_svm, target_names=["non_autistic", "autistic"], output_dict=True
    )
    rf_report = classification_report(
        y, y_pred_rf, target_names=["non_autistic", "autistic"], output_dict=True
    )

    return {"svm": svm_report, "random_forest": rf_report}


# ---------------------------------------------------
# Classificar nova imagem
# ---------------------------------------------------
def classify_image(image_path):
    """
    Classifica uma nova imagem como autista ou não autista.

    Args:
        image_path (str): Caminho para a imagem.

    Retorna:
        dict com predições da SVM e Random Forest.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Carregar modelos e pré-processadores
    svm_model = joblib.load("TrainSVM_RF/svm_model.pkl")
    rf_model = joblib.load("TrainSVM_RF/rf_model.pkl")
    scaler = joblib.load("TrainSVM_RF/scaler.pkl")
    imputer = joblib.load("TrainSVM_RF/imputer.pkl")  # Carregar o imputer também

    # Extrair AUs com o Detector
    detector = Detector()
    preds = detector.detect_image(image_path)

    if preds.empty:
        raise ValueError("Nenhum rosto detectado na imagem!")

    # Extrair apenas as colunas das AUs
    aus = preds.iloc[0].filter(regex="^AU").to_dict()
    X_new = pd.DataFrame([aus])
    
    # Aplicar o mesmo pré-processamento do treinamento
    X_imputed = imputer.transform(X_new)  # Tratar valores NaN
    X_scaled = scaler.transform(X_imputed)  # Normalizar

    # Predições
    svm_pred = svm_model.predict(X_scaled)[0]
    rf_pred = rf_model.predict(X_scaled)[0]

    svm_label = "autistic" if svm_pred == 1 else "non_autistic"
    rf_label = "autistic" if rf_pred == 1 else "non_autistic"

    return {
        "image": image_path,
        "svm_prediction": svm_label,
        "rf_prediction": rf_label,
        "aus_values": {
            au: f"{val:.2f} ({au_descriptions.get(au, 'Desconhecido')})"
            for au, val in aus.items()
        }
    }


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Avaliar ou classificar imagens com SVM/Random Forest")
    parser.add_argument("--csv", type=str, help="Avaliar dataset CSV")
    parser.add_argument("--image", type=str, help="Classificar uma nova imagem")

    args = parser.parse_args()

    if args.csv:
        reports = evaluate_models(args.csv)
        print("\n📊 Avaliação no dataset:")
        print("SVM Report:")
        print(pd.DataFrame(reports["svm"]).round(3))
        print("\nRandom Forest Report:")
        print(pd.DataFrame(reports["random_forest"]).round(3))

    if args.image:
        result = classify_image(args.image)
        print("\n🔍 Classificação da imagem:")
        print(f"Imagem: {result['image']}")
        print(f"Predição SVM: {result['svm_prediction']}")
        print(f"Predição Random Forest: {result['rf_prediction']}")
        print("\nValores das Action Units:")
        for au, desc in result['aus_values'].items():
            print(f"  {au}: {desc}")


if __name__ == "__main__":
    main()