# Reconhecimento de AUs (Facial Action Units) com Py-Feat

Este projeto utiliza a biblioteca [py-feat](https://py-feat.org/) para detectar **Action Units (AUs)** faciais em imagens.  
As AUs fazem parte do **FACS (Facial Action Coding System)**, um sistema que descreve movimentos musculares do rosto e é muito usado em análise de emoções, psicologia e visão computacional.

---

## 🚀 Funcionalidades

- Detecta **rostos** em imagens.
- Extrai **Action Units (AUs)** associadas a movimentos faciais.
- Exibe os resultados no terminal em formato legível (AU + descrição + intensidade).
- Destaca a **AU dominante** na imagem exibida.
- Utiliza um dicionário externo (`au_descriptions.json`) com as descrições das AUs em **português**.

---

## 🔧 Instalação

### 1. Criar e ativar ambiente virtual

No Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

No Linux/Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Instalar dependências

O arquivo `requirements.txt` já contém as dependências necessárias.
Basta rodar:

```bash
pip install -r requirements.txt
```

---

## ▶️ Como rodar o código

Na raiz do projeto (`TEST CODES`), execute:

```bash
python Aus/detect_aus_image.py
```

* O script vai processar a imagem em `Aus/PhotoForTesting.jpg`.
* Ele imprime no **terminal** as AUs detectadas.

---

## 📊 Exemplo de saída

No terminal:

```
Resultados das Action Units na imagem:
AU01 (Elevacao da sobrancelha interna): 0.32
AU06 (Contracao da bochecha): 0.91
AU07 (Tensao da palpebra): 1.00
AU12 (Sorriso - elevacao do canto da boca): 0.84
...
```
---

## ⬇️ Códgio
[Clique aqui.](https://github.com/LuiisMarim/VITA-Documentation/blob/main/VITA/Test%20Codes/Aus/detect_aus_image.py)

---

## 📚 Referências

* **FACS (Facial Action Coding System)**: Paul Ekman & Wallace V. Friesen.
* [py-feat Documentation](https://py-feat.org/)

