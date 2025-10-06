# Célula 2: Carregamento e Análise Exploratória
# Carregar dados
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

print("📊 Dimensões do dataset:", df.shape)
print("\n🔍 Primeiras linhas:")
print(df.head())

print("\n📈 Estatísticas descritivas:")
print(df.describe())

# Análise exploratória visual
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribuição do preço
axes[0,0].hist(df['PRICE'], bins=50, alpha=0.7, color='skyblue')
axes[0,0].set_title('Distribuição dos Preços dos Imóveis')
axes[0,0].set_xlabel('Preço (em $100,000)')
axes[0,0].set_ylabel('Frequência')

# Correlação com MedInc
axes[0,1].scatter(df['MedInc'], df['PRICE'], alpha=0.3)
axes[0,1].set_title('Renda Média vs Preço do Imóvel')
axes[0,1].set_xlabel('Renda Média')
axes[0,1].set_ylabel('Preço')

# Correlação com Age
axes[1,0].scatter(df['HouseAge'], df['PRICE'], alpha=0.3, color='green')
axes[1,0].set_title('Idade da Casa vs Preço')
axes[1,0].set_xlabel('Idade da Casa')
axes[1,0].set_ylabel('Preço')

# Mapa de correlações
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', ax=axes[1,1], cmap='coolwarm')
axes[1,1].set_title('Mapa de Correlação')

plt.tight_layout()
plt.show()