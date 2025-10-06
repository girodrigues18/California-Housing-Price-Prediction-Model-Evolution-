# Célula 3: Preparação dos Dados e Baseline
# Dividir dados
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("📋 Divisão dos dados:")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# 🎯 MODELO 1: Baseline - Regressão Linear
print("\n" + "="*50)
print("🎯 MODELO 1: Regressão Linear (Baseline)")
print("="*50)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"📊 Performance da Baseline:")
print(f"Erro Quadrático Médio (MSE): {mse_baseline:.4f}")
print(f"Coeficiente R²: {r2_baseline:.4f}")
print(f"💡 Erro aproximado em dólares: ${np.sqrt(mse_baseline)*100000:,.0f}")

# Visualizar resultados da baseline
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_baseline, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Regressão Linear: Reais vs Preditos')

plt.subplot(1, 2, 2)
residuos = y_test - y_pred_baseline
plt.scatter(y_pred_baseline, residuos, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')

plt.tight_layout()
plt.show()

print("🔍 Diagnóstico: Modelo mostra underfitting - muito simples para a complexidade dos dados")