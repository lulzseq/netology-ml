param_grid = [{
	'C': np.linspace(0.001,1,10, dtype=float),
	'penalty': ['l1', 'l2'],
	'solver': ['liblinear']
}]

logreg = LogisticRegression(n_jobs=-1)

gs_log = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='accuracy', cv=10)
gs_log = gs_log.fit(X_train_std, y_train)

with open('../models/logistic_regression.pickle', 'wb') as f:
	pickle.dump(gs_log, f)

with open('../reports/models.txt', 'a') as f:
	f.write('Model: {}\n'.format(str(logreg).split('(')[0]))
	f.write('Accuracy: {}\n'.format(gs_log.best_score_))
	f.write('Best params: {}'.format(gs_log.best_params_))
	f.write('\n---\n')
f.close()