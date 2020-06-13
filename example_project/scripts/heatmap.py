corr_values = df.iloc[:,1:]
plt.figure(figsize=(12,8), dpi=80)
sns.heatmap(corr_values.corr(), yticklabels=corr_values.columns, cbar=True, cmap='coolwarm', annot=True)
plt.savefig('../reports/heatmap.png')