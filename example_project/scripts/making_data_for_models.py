#add column 'quality_rate'
def quality_rate(x):
	if x['quality'] > 6:
		res = 1
	else:
		res = 0
	return res

df['quality_rate'] = df.apply(quality_rate, axis=1)
df.to_csv('../data/intermid/df-quality-rate.csv')

with open('../data/intermid/description.txt', 'a') as f:
	f.write("— add 'quality_rate' column. Output file: 'df-quality-rate.csv'.\n")
f.close()

#train_test_split
X = df.iloc[:, 1:12]
y = df['quality_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('../data/intermid/X_train.csv')
X_test.to_csv('../data/intermid/X_test.csv')
y_train.to_csv('../data/intermid/y_train.csv')
y_test.to_csv('../data/intermid/y_test.csv')

#scaling
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

np.savetxt('../data/intermid/X_train_std.csv', X_train_std)
np.savetxt('../data/intermid/X_test_std.csv', X_test_std)

with open('../data/intermid/description.txt', 'a') as f:
	f.write("— using StandardScaler for X_train and X_test. Output file: 'X_train_std' and 'X_test_std'.\n")
f.close()