param_grid = {
	'max_depth': np.linspace(1,50,10, dtype=int),
	'n_estimators': np.linspace(1,100,10, dtype=int)
}

xgb = xgb.XGBClassifier(objective = 'binary:logistic', n_jobs=-1)

gs_xgb = GridSearchCV(estimator = xgb, param_grid = param_grid, scoring = make_scorer(accuracy_score), cv = 10, refit = 'accuracy_score')
gs_xgb.fit(X_train, y_train)

with open('../models/xgboost.pickle', 'wb') as f:
	pickle.dump(gs_xgb, f)

with open('../reports/models.txt', 'a') as f:
	f.write('Model: {}\n'.format(str(xgb).split('(')[0]))
	f.write('Accuracy: {}\n'.format(gs_xgb.best_score_))
	f.write('Best params: {}'.format(gs_xgb.best_params_))
	f.write('\n---\n')
f.close()