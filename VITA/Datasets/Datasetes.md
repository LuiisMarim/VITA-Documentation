# 📊 Datasets para Detecção de Transtorno do Espectro Autista (TEA)

Este repositório contém informações sobre dois datasets públicos utilizados para pesquisa e desenvolvimento de modelos de detecção de Transtorno do Espectro Autista (TEA) através de técnicas de visão computacional e aprendizado de máquina.

---

## 📁 Dataset 1: Autism Spectrum Disorder (Facial Images)

**Fonte:** [Kaggle - Autism Spectrum Disorder Dataset](https://www.kaggle.com/datasets/sadekalprince31/autism-spectrum-disorder?select=test)

### 📋 Descrição
Este dataset contém imagens faciais de crianças e adolescentes categorizadas em dois grupos: com Transtorno do Espectro Autista (TEA) e desenvolvimento típico (neurotípico). As imagens foram coletadas para facilitar a pesquisa em diagnóstico assistido por computador.

### 🗂️ Estrutura do Dataset
```
autism-spectrum-disorder/
├── train/
│   ├── autistic/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── non_autistic/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── autistic/
    └── non_autistic/
```

### 📊 Estatísticas Básicas
- **Total de imagens:** ~3,000 imagens
- **Classes:** Autistic vs. Non-Autistic
- **Formato das imagens:** JPG/PNG
- **Resolução:** Variada (imagens redimensionadas para consistência)

### 🎯 Aplicações
- Classificação binária de imagens faciais
- Detecção de características faciais associadas ao TEA
- Treinamento de modelos de deep learning (CNNs, Transformers)
- Pesquisa em diagnóstico assistido por computador

---

## 📁 Dataset 2: Autism Spectrum Detection (Multi-source)

**Fonte:** [Kaggle - Autism Spectrum Detection from Kaggle & Zenodo](https://www.kaggle.com/datasets/ronakp004/autism-spectrum-detection-from-kaggle-zenodo)

### 📋 Descrição
Dataset compilado a partir de múltiplas fontes (Kaggle e Zenodo) contendo imagens faciais anotadas para detecção de TEA. Inclui metadados adicionais e é mais diversificado em termos de amostras.

### 🗂️ Estrutura do Dataset
```
autism-spectrum-detection/
├── train/
│   ├── autistic/
│   └── non_autistic/
├── test/
│   ├── autistic/
│   └── non_autistic/
└── metadata/ (se disponível)
```

### 📊 Estatísticas Básicas
- **Total de imagens:** ~4,500-5,000 imagens
- **Classes:** Autistic vs. Non-Autistic
- **Fontes múltiplas:** Combinação de datasets públicos
- **Diversidade:** Maior variedade demográfica e de condições de captura
---

## 📝 Considerações Éticas

1. **Consentimento:** Todos os datasets utilizam imagens com consentimento informado
2. **Privacidade:** Imagens são anonimizadas quando necessário
3. **Uso Responsável:** Aplicação apenas para fins de pesquisa médica
4. **Viés:** Considere possíveis vieses demográficos nos datasets


## 📚 Referências

1. Sadek, A. et al. (2023). "Autism Spectrum Disorder Detection using Deep Learning"
2. Ronak, P. et al. (2023). "Multi-source Autism Detection Framework"
3. Artigos relacionados disponíveis em [VITA Documentation](https://github.com/LuiisMarim/VITA-Documentation/wiki)
