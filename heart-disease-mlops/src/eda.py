import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ucimlrepo import fetch_ucirepo

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts/plots", exist_ok=True)

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)
df.rename(columns={"num": "target"}, inplace=True)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("data/heart.csv", index=False)

# Class balance
plt.figure()
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.savefig("artifacts/plots/class_balance.png")
plt.close()

# Histograms
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig("artifacts/plots/histograms.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.savefig("artifacts/plots/correlation.png")
plt.close()

print("EDA completed successfully")
