gs_list = [gs_log, gs_tree, gs_xgb]

for gs in gs_list:
	y_pred = gs.best_estimator_.predict(X_test)
	print(f"Model: {gs}")
	print('Accuracy: %.2f' % accuracy_score(y_true=y_test, y_pred=y_pred))
	print('Precision: %.2f' % precision_score(y_true=y_test, y_pred=y_pred))
	print('Recall: %.2f' % recall_score(y_true=y_test, y_pred=y_pred))
	print('f1_score: %.2f' % f1_score(y_true=y_test, y_pred=y_pred))
	print()

	with open('../reports/score_models.txt', 'a') as f:
		f.write('Model: {}\n'.format(gs))
		f.write('Accuracy: {:.2f}\n'.format(accuracy_score(y_true=y_test, y_pred=y_pred)))
		f.write('Precision: {:.2f}\n'.format(precision_score(y_true=y_test, y_pred=y_pred)))
		f.write('Recall: {:.2f}\n'.format(recall_score(y_true=y_test, y_pred=y_pred)))
		f.write('f1_score: {:.2f}\n'.format(f1_score(y_true=y_test, y_pred=y_pred)))
		f.write('---\n')
	f.close()