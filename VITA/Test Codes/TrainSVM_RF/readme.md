# Módulos de Treinamento e Avaliação de Modelos de Classificação

Este diretório contém os scripts para treinamento e avaliação de modelos de machine learning para classificação baseada em Action Units (AUs).

## 📋 Visão Geral dos Módulos

### 1. train_models.py
Script responsável pelo treinamento dos modelos de classificação utilizando SVM e Random Forest.

### 2. evaluate_models.py
Script para avaliação dos modelos treinados e classificação de novas imagens.

---

## 🗂️ Estrutura de Arquivos

```
📁 Train/
├── train_models.py          # Script de treinamento dos modelos
├── evaluate_models.py       # Script de avaliação e predição
├── svm_model.pkl           # Modelo SVM treinado (output)
├── rf_model.pkl            # Modelo Random Forest treinado (output)
├── scaler.pkl              # Scaler para normalização (output)
└── imputer.pkl             # Imputer para tratamento de dados (output)
```

---

## 🔧 Pré-requisitos

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

### Arquivos Necessários
- `../BuildDataset/dataset_faus.csv` - Dataset com features extraídas
- `../Aus/au_descriptions.json` - Descrições das Action Units

---

# train_models.py

## 🎯 Funcionalidades

Treina dois modelos de classificação para análise de expressões faciais:
- Support Vector Machine (SVM) com kernel RBF
- Random Forest com 200 estimadores

## ⚙️ Fluxo de Processamento

1. **Carregamento do Dataset**: Lê o arquivo CSV com as features extraídas
2. **Pré-processamento**:
   - Tratamento de valores faltantes com SimpleImputer
   - Normalização dos dados com StandardScaler
3. **Divisão dos Dados**: Split 80/20 com estratificação por classe
4. **Treinamento**: Treina ambos os modelos simultaneamente
5. **Salvamento**: Exporta modelos e pré-processadores em formato pickle

## ▶️ Como Executar

```bash
python train_models.py
```

## 📊 Saída Esperada

```
Carregando dataset...
Colunas encontradas: ['id', 'label', 'AU01', 'AU02', ...]
Total de amostras: 150
Distribuição das labels: autistic 75, non_autistic 75
Valores NaN por coluna: AU01 0, AU02 0, ...
Treinando SVM...
Treinando Random Forest...
Acurácia SVM (treino): 0.9500
Acurácia Random Forest (treino): 0.9800
✅ Modelos treinados e salvos com sucesso!
📁 Arquivos salvos: svm_model.pkl, rf_model.pkl, scaler.pkl, imputer.pkl
```

---

# evaluate_models.py

## 🎯 Funcionalidades

Oferece dois modos de operação:
1. **Modo Avaliação**: Avalia modelos no dataset completo
2. **Modo Predição**: Classifica uma nova imagem individual

## ⚙️ Características

- Interface CLI com argparse
- Compatível com o pré-processamento do treinamento
- Fornece descrições das AUs detectadas
- Suporte a ambos os modelos (SVM e Random Forest)

## ▶️ Como Executar

### Modo Avaliação (dataset completo)
```bash
python evaluate_models.py --csv ../BuildDataset/dataset_faus.csv
```

### Modo Predição (imagem individual)
```bash
python evaluate_models.py --image caminho/para/imagem.jpg
```

## 📊 Saída Esperada - Modo Avaliação

```
📊 Avaliação no dataset:
SVM Report:
              precision  recall  f1-score  support
non_autistic     0.950    0.920     0.935      50
autistic         0.925    0.955     0.940      50

Random Forest Report:
              precision  recall  f1-score  support
non_autistic     0.980    0.960     0.970      50
autistic         0.961    0.980     0.970      50
```

## 📊 Saída Esperada - Modo Predição

```
🔍 Classificação da imagem:
Imagem: caminho/para/imagem.jpg
Predição SVM: autistic
Predição Random Forest: autistic

Valores das Action Units:
  AU01: 0.75 (Elevação da sobrancelha interna)
  AU02: 0.32 (Elevação da sobrancelha externa)
  AU04: 0.89 (Abaixamento das sobrancelhas)
  ...
```

---

## ⚠️ Observações Importantes

1. **Dependências**: Os scripts dependem do dataset gerado pelo `build_dataset.py`
2. **Compatibilidade**: Usa os mesmos pré-processadores do treinamento nas predições
3. **Formato de Imagens**: Suporte para JPG, JPEG e PNG no modo predição
4. **Metadados**: Requer o arquivo `au_descriptions.json` para descrições das AUs

---

## 📚 Referências

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Py-Feat Documentation](https://py-feat.org/)
- [Joblib Documentation](https://joblib.readthedocs.io/)

---

## 📄 Códigos

- [train_models.py](https://github.com/LuiisMarim/VITA-Documentation/blob/main/VITA/Test%20Codes/Train/train_models.py)
- [evaluate_models.py](https://github.com/LuiisMarim/VITA-Documentation/blob/main/VITA/Test%20Codes/Train/evaluate_models.py)


⚠️ OBS: Ambas as imagens referentes a pasta "TrainSVM_RF" são de individuos com TEA.