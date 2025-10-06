# Célula 5: Feature Importance
print("\n" + "="*50)
print("🎯 ANÁLISE: Feature Importance com Random Forest")
print("="*50)

# Modelo rápido para análise de importância
rf_importance = RandomForestRegressor(n_estimators=100, random_state=42)
rf_importance.fit(X_train, y_train)

# Obter importância das features
importances = rf_importance.feature_importances_
feature_names = california.feature_names

# Criar DataFrame para visualização
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=True)

# Plotar importância
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'], color='lightcoral')
plt.xlabel('Importância')
plt.title('Importância das Variáveis (Random Forest)')
plt.tight_layout()
plt.show()

print("💡 Insight: 'MedInc' (Renda Média) é a variável mais importante para prever preços")