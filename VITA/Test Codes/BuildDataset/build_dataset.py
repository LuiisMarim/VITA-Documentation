import os
import json
import pandas as pd
from feat import Detector

# ---------------------------------------------------
# Definir caminho base
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(BASE_DIR, "..", "Aus", "au_descriptions.json")
autistic_path = os.path.join(BASE_DIR, "..", "BuildDataset", "autistic")
non_autistic_path = os.path.join(BASE_DIR, "..", "BuildDataset", "non_autistic")

# ---------------------------------------------------
# Carregar dicionário de AUs a partir do JSON externo
# ---------------------------------------------------
with open(json_path, "r", encoding="utf-8") as f:
    au_descriptions = json.load(f)

# ---------------------------------------------------
# Inicialização do detector
# ---------------------------------------------------
detector = Detector()

# ---------------------------------------------------
# Definir as pastas de entrada (classes)
# ---------------------------------------------------
folders = {
    "autistic": autistic_path,          # Pasta com imagens de crianças autistas
    "non_autistic": non_autistic_path   # Pasta com imagens de crianças não autistas
}

# Lista para acumular resultados
results = []

# ---------------------------------------------------
# Loop em cada pasta e processar imagens
# ---------------------------------------------------
for label, folder in folders.items():
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)

        # Ignorar arquivos que não são imagens
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            # Detectar Action Units (AUs) na imagem
            preds = detector.detect_image(img_path)

            # Selecionar apenas colunas de AUs
            au_columns = [col for col in preds.columns if col.startswith("AU")]
            aus = preds[au_columns].iloc[0].to_dict()

            # Montar linha com metadados + valores
            row = {"id": img_file, "label": label}
            row.update(aus)

            results.append(row)

            print(f"✅ Processado: {img_file} ({label})")

        except Exception as e:
            print(f"⚠️ Erro ao processar {img_file}: {e}")

# ---------------------------------------------------
# Construir DataFrame final e salvar CSV
# ---------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("dataset_faus.csv", index=False, encoding="utf-8")

print("\n🎉 Dataset gerado com sucesso: dataset_faus.csv")
print(f"Total de amostras: {len(df)}")
