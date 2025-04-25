import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Summary Statistics
print(df.describe(include='all'))
print(df.isnull().sum())

# Histograms
df.hist(column=['Age', 'Fare'], bins=20, figsize=(12, 5))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# Boxplots
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age and Fare")
plt.tight_layout()
plt.savefig("boxplot_eda.png")
plt.close()

# Correlation Heatmap
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()
