import json
import cv2
from feat import Detector

# ---------------------------------------------------
# Carregar dicionário de AUs a partir do JSON externo
# ---------------------------------------------------
with open("Aus/au_descriptions.json", "r", encoding="utf-8") as f:
    au_descriptions = json.load(f)


# ---------------------------------------------------
# Inicialização do detector
# ---------------------------------------------------
detector = Detector()

# ---------------------------------------------------
# Caminho da imagem a ser processada
# ---------------------------------------------------
image_path = "Aus/PhotoForTesting.jpg"

# ---------------------------------------------------
# Detectar Action Units na imagem
# ---------------------------------------------------
preds = detector.detect_image(image_path)

# Obtemos apenas os valores das AUs (colunas que começam com "AU")
au_columns = [col for col in preds.columns if col.startswith("AU")]
aus = preds[au_columns].iloc[0].to_dict()

# ---------------------------------------------------
# Construir saída formatada
# ---------------------------------------------------
formatted = {}
for au, val in aus.items():
    desc = au_descriptions.get(au, "Desconhecido")
    formatted[au] = f"{au} ({desc}): {val:.2f}"

print("Resultados das Action Units na imagem:")
for au, text in formatted.items():
    print(text)

# ---------------------------------------------------
# Mostrar imagem com AU dominante
# ---------------------------------------------------
frame = cv2.imread(image_path)

top_au = max(aus, key=aus.get)
desc_top = au_descriptions.get(top_au, "Desconhecido")

cv2.putText(frame, f"{top_au}: {desc_top}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.waitKey(0)
cv2.destroyAllWindows()
