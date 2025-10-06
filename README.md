# 🏠 California Housing Price Prediction - Model Evolution
Um projeto completo de Machine Learning supervisionado demonstrando a evolução de modelos de regressão para previsão de preços de imóveis na Califórnia. Do baseline simples às técnicas avançadas de ensemble learning.

## 📊 Sobre o Projeto

Este projeto aborda o problema clássico de previsão de preços de imóveis usando o dataset California Housing. A abordagem principal é demonstrar a evolução metodológica desde modelos simples até técnicas avançadas, mostrando como cada melhoria impacta a performance preditiva.

**Problema de Negócio**: Prever valores de imóveis para auxiliar em avaliações imobiliárias, investimentos e análise de mercado.

**Abordagem Técnica**: Regressão supervisionada com comparação sistemática de múltiplos algoritmos.

## 🎯 Objetivos

### Principal
- Desenvolver e comparar diferentes modelos de machine learning para previsão de preços de imóveis

### Específicos
- Estabelecer uma baseline com Regressão Linear
- Identificar e tratar underfitting através de modelos mais complexos
- Otimizar hiperparâmetros usando GridSearchCV
- Implementar e comparar técnicas de ensemble learning
- Analisar a importância das features no modelo final

## 🔧 Metodologia

### 1. Análise Exploratória (EDA)
- Estatísticas descritivas
- Matriz de correlação
- Distribuição das variáveis
- Identificação de relações entre features e target

### 2. Pré-processamento
- Divisão train/test (70%/30%)
- Validação cruzada para otimização
- Nenhum tratamento de outliers (para manter comparabilidade)

### 3. Modelos Implementados

#### 🎯 Baseline
- **Regressão Linear**: Modelo simples para estabelecer baseline de performance

#### ⚙️ Modelos Otimizados
- **Decision Tree Regressor**: Com GridSearchCV para otimização de hiperparâmetros
- **Random Forest**: Ensemble method (Bagging) com 200 estimadores
- **Gradient Boosting**: Ensemble method (Boosting) com 200 estimadores

### 4. Avaliação de Modelos
**Métricas Principais**:
- `MSE` (Mean Squared Error): Erro quadrático médio
- `R²` (Coefficient of Determination): Variância explicada

## 💡 Principais Aprendizados

### Técnicos
- **Underfitting:** Modelos simples podem não capturar complexidades dos dados
- **Ensemble Learning:** Combinação de modelos melhora robustez e performance
- **Otimização:** GridSearch é essencial para extrair máximo performance
- **Feature Analysis:** Entender importância das variáveis guia melhorias

### Metodológicos
- **Iteração:** Abordagem incremental gera melhores resultados
- **Baseline:** Sempre estabelecer referência para medir progresso
- **Visualização:** Gráficos são cruciais para comunicar insights

## 🎯 Conclusões
- Gradient Boosting apresentou a melhor performance geral
- Técnicas de ensemble learning superaram significativamente modelos individuais
- Renda média é o fator mais determinante para preços de imóveis
- A evolução metodológica mostrou ganhos progressivos de performance
