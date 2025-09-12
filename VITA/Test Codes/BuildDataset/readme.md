# Construção de Dataset de Action Units (AUs) para Análise Facial

Este script Python é responsável por construir um dataset contendo Action Units (AUs) faciais detectadas em imagens de crianças autistas e não autistas, utilizando a biblioteca Py-Feat.

## 🎯 Objetivo

Criar um dataset estruturado (`dataset_faus.csv`) com as seguintes informações:
- Identificação da imagem
- Rótulo (autistic/non_autistic)
- Valores de intensidade para cada Action Unit (AU) detectada

---

## 📋 Funcionalidades

- Processa imagens de duas categorias: autistas e não autistas
- Detecta automaticamente Action Units (AUs) utilizando Py-Feat
- Combina metadados das imagens com os valores das AUs detectadas
- Exporta os resultados para um arquivo CSV estruturado
- Trata exceções e fornece feedback durante o processamento

---

## 🗂️ Estrutura de Diretórios

O script espera a seguinte estrutura de pastas:
```
BuildDataset/
├── autistic/          # Imagens de crianças autistas
└── non_autistic/      # Imagens de crianças não autistas
```

---

## 🔧 Instalação e Configuração

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

### 2. Preparar estrutura de diretórios
Certifique-se de que as pastas `autistic` e `non_autistic` existam no diretório `BuildDataset` com as imagens apropriadas.

### 3. Arquivo de descrição das AUs
O script utiliza um arquivo JSON (`au_descriptions.json`) localizado em `../Aus/` que contém as descrições das Action Units.

---

## ▶️ Como Executar

Execute o script a partir do diretório onde ele está localizado:

```bash
python build_dataset.py
```

---

## 📊 Saída Esperada

O script gerará:
- Feedback no terminal sobre o processamento de cada imagem
- Um arquivo `dataset_faus.csv` contendo:
  - Coluna `id`: nome do arquivo de imagem
  - Coluna `label`: categoria (autistic/non_autistic)
  - Colunas AU##: valores de intensidade para cada Action Unit detectada

Exemplo de saída no terminal:
```
✅ Processado: image1.jpg (autistic)
✅ Processado: image2.jpg (non_autistic)
⚠️ Erro ao processar image3.jpg: [mensagem de erro]

🎉 Dataset gerado com sucesso: dataset_faus.csv
Total de amostras: 150
```

---

## ⚠️ Tratamento de Erros

O script inclui tratamento de exceções que:
- Ignora arquivos não-imagem (apenas processa .jpg, .jpeg, .png)
- Captura e exibe erros durante o processamento das imagens
- Continua executando mesmo quando ocorrem erros em imagens específicas

---

## 📚 Referências

- [Py-Feat Documentation](https://py-feat.org/)
- Facial Action Coding System (FACS) - Paul Ekman & Wallace V. Friesen
- [Arquivo de descrição de AUs](https://github.com/LuiisMarim/VITA-Documentation/blob/main/VITA/Test%20Codes/Aus/au_descriptions.json)

---

## 📄 Código

[Clique aqui para visualizar o código completo](https://github.com/LuiisMarim/VITA-Documentation/blob/main/VITA/Test%20Codes/BuildDataset/build_dataset.py)