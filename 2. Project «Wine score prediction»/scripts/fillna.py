na_feature_list = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'pH', 'sulphates']

for na_ftr in na_feature_list:
	df[na_ftr] = df[na_ftr].fillna(int(df[na_ftr].median()))

df.to_csv('../data/intermid/df-fillna.csv')