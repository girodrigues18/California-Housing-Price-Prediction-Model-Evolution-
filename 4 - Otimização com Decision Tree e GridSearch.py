# Célula 4: Otimização com Decision Tree e GridSearch
print("\n" + "="*50)
print("🔧 MODELO 2: Decision Tree com GridSearch")
print("="*50)

# Definir parâmetros para otimização
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Buscar melhores parâmetros
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"🎯 Melhores parâmetros: {grid_search.best_params_}")

# Treinar com melhores parâmetros
best_tree = grid_search.best_estimator_
y_pred_tree = best_tree.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"📊 Performance da Decision Tree Otimizada:")
print(f"Erro Quadrático Médio (MSE): {mse_tree:.4f}")
print(f"Coeficiente R²: {r2_tree:.4f}")
print(f"📈 Melhoria em relação à baseline: {((mse_baseline - mse_tree)/mse_baseline)*100:.1f}%")