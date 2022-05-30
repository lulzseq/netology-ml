param_grid = [{
	'max_depth': np.linspace(1,20,5, dtype=int),
	'criterion': ['entropy', 'gini', 'error'],
	'max_leaf_nodes': np.linspace(1,30,10, dtype=int)
}]

tree = DecisionTreeClassifier()

gs_tree = GridSearchCV(estimator=tree, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs_tree = gs_tree.fit(X_train, y_train)

with open('../models/decision_tree.pickle', 'wb') as f:
	pickle.dump(gs_tree, f)

with open('../reports/models.txt', 'a') as f:
	f.write('Model: {}\n'.format(str(tree).split('(')[0]))
	f.write('Accuracy: {}\n'.format(gs_tree.best_score_))
	f.write('Best params: {}'.format(gs_tree.best_params_))
	f.write('\n---\n')
f.close()