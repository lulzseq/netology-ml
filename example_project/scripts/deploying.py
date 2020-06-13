y_pred = gs_xgb.best_estimator_.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred.to_csv('../data/precessed/y_pred.csv')

df['y_pred'] = y_pred
df.to_csv('../data/precessed/wine_prediction_final.csv', index = False )