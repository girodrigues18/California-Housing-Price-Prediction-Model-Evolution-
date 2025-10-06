# Célula 7: Comparação Final e Visualização
print("\n" + "="*50)
print("📊 COMPARAÇÃO FINAL: Todos os Modelos")
print("="*50)

# Criar DataFrame comparativo
comparison = pd.DataFrame({
    'Modelo': ['Regressão Linear', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'MSE': [mse_baseline, mse_tree, mse_rf, mse_gb],
    'R²': [r2_baseline, r2_tree, r2_rf, r2_gb]
})

print(comparison.round(4))

# Visualização comparativa
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
model_predictions = [y_pred_baseline, y_pred_tree, y_pred_rf, y_pred_gb]
model_names = ['Regressão Linear', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
colors = ['blue', 'green', 'orange', 'red']

for i, (ax, pred, name, color) in enumerate(zip(axes.flat, model_predictions, model_names, colors)):
    ax.scatter(y_test, pred, alpha=0.3, color=color)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Preditos')
    ax.set_title(f'{name}\nMSE: {mean_squared_error(y_test, pred):.4f}')
    
    # Adicionar R² no gráfico
    r2 = r2_score(y_test, pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"🎉 Melhor modelo: Gradient Boosting com {((mse_baseline - mse_gb)/mse_baseline)*100:.1f}% de melhoria no MSE!")