"""
Treinamento de modelos SVM e Random Forest para análise de expressões faciais
em crianças autistas e não autistas a partir de Action Units (AUs).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------
# Definir caminho base do projeto
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv = os.path.join(BASE_DIR, "..", "BuildDataset", "dataset_faus.csv")

# ---------------------------------------------------
# Função principal para treinamento dos modelos
# ---------------------------------------------------
def train_and_save_models(csv_path=csv):
    """
    Lê o dataset de AUs, treina SVM e Random Forest e salva os modelos.

    Args:
        csv_path (str): Caminho para o dataset em formato CSV.

    Saída:
        - svm_model.pkl: Modelo SVM treinado
        - rf_model.pkl: Modelo Random Forest treinado
        - scaler.pkl: Scaler para normalização dos dados
        - imputer.pkl: Imputer para tratamento de valores faltantes
    """
    
    # ---------------------------------------------------
    # Carregamento e inspeção do dataset
    # ---------------------------------------------------
    print("Carregando dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Colunas encontradas: {df.columns.tolist()}")
    print(f"Total de amostras: {len(df)}")
    print(f"Distribuição das labels: {df['label'].value_counts()}")

    # ---------------------------------------------------
    # Verificação e tratamento de valores NaN
    # ---------------------------------------------------
    print(f"Valores NaN por coluna:\n{df.isnull().sum()}")
    
    # ---------------------------------------------------
    # Separação entre features e labels
    # ---------------------------------------------------
    X = df.drop(columns=["id", "label"])
    y = df["label"].map({"autistic": 1, "non_autistic": 0})

    print(f"Labels únicos após mapeamento: {y.unique()}")
    print(f"Distribuição das classes: {y.value_counts()}")

    # ---------------------------------------------------
    # Tratamento de valores faltantes com SimpleImputer
    # ---------------------------------------------------
    print("Tratando valores NaN...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    print(f"Valores NaN após imputação: {np.isnan(X_imputed).sum()}")

    # ---------------------------------------------------
    # Normalização dos dados com StandardScaler
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # ---------------------------------------------------
    # Divisão dos dados em conjuntos de treino e teste
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Tamanho do conjunto de treino: {X_train.shape}")
    print(f"Tamanho do conjunto de teste: {X_test.shape}")

    # ---------------------------------------------------
    # Treinamento do modelo SVM
    # ---------------------------------------------------
    print("Treinando SVM...")
    svm_model = SVC(
        kernel="rbf", 
        C=1.0, 
        gamma="scale", 
        probability=True, 
        random_state=42
    )
    svm_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Treinamento do modelo Random Forest
    # ---------------------------------------------------
    print("Treinando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Avaliação dos modelos nos dados de treino
    # ---------------------------------------------------
    svm_score = svm_model.score(X_train, y_train)
    rf_score = rf_model.score(X_train, y_train)
    print(f"Acurácia SVM (treino): {svm_score:.4f}")
    print(f"Acurácia Random Forest (treino): {rf_score:.4f}")

    # ---------------------------------------------------
    # Salvamento dos modelos e pré-processadores
    # ---------------------------------------------------
    joblib.dump(svm_model, "TrainSVM_RF/svm_model.pkl")
    joblib.dump(rf_model, "TrainSVM_RF/rf_model.pkl")
    joblib.dump(scaler, "TrainSVM_RF/scaler.pkl")
    joblib.dump(imputer, "TrainSVM_RF/imputer.pkl")

    print("✅ Modelos treinados e salvos com sucesso!")
    print("📁 Arquivos salvos: svm_model.pkl, rf_model.pkl, scaler.pkl, imputer.pkl")

# ---------------------------------------------------
# Ponto de entrada principal do script
# ---------------------------------------------------
if __name__ == "__main__":
    train_and_save_models(csv)