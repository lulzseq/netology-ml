df.describe().to_csv('../reports/df-describe.csv')
f = open('../reports/df-info.txt', 'w+')
df.info(buf=f)
f.close()