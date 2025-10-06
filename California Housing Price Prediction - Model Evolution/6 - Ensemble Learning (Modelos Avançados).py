# Célula 6: Ensemble Learning - Modelos Avançados
print("\n" + "="*50)
print("🌟 MODELO 3: Ensemble Learning")
print("="*50)

# 🎯 Random Forest
print("🌲 Treinando Random Forest (Bagging)...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 🎯 Gradient Boosting
print("🚀 Treinando Gradient Boosting (Boosting)...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Avaliar modelos
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"📊 Performance dos Modelos Ensemble:")
print(f"Random Forest    - MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")
print(f"Gradient Boosting - MSE: {mse_gb:.4f}, R²: {r2_gb:.4f}")