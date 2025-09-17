# 📘 Data Analysis & Visualization Assignment
# Author: [Sholedolu Aminat Omowunmi]
# Date: [17-09-2025]

# ==============================
# Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ==============================
# Task 1: Load and Explore Dataset
# ==============================
try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # Convert to pandas DataFrame
    
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Show first few rows
print("\n📌 First 5 Rows of the Dataset:")
display(df.head())

# Check data types and missing values
print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 Missing Values:")
print(df.isnull().sum())

# Clean dataset (fill/drop missing values if any)
df = df.dropna()
print("\n✅ Cleaned dataset (missing values removed if present).")

# ==============================
# Task 2: Basic Data Analysis
# ==============================

# Descriptive statistics
print("\n📊 Basic Statistics:")
display(df.describe())

# Grouping example: Mean of numerical columns by species
grouped = df.groupby("target").mean()
print("\n📊 Mean values grouped by species:")
display(grouped)

# Interesting finding: correlation matrix
print("\n📌 Correlation Matrix:")
display(df.corr())

# ==============================
# Task 3: Data Visualizations
# ==============================

# 1. Line Chart (simulate a trend using first feature vs index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart - Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x="target", y="petal length (cm)", data=df, estimator="mean")
plt.title("Bar Chart - Average Petal Length per Species")
plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram - Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (Sepal Length vs Petal Length)
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="deep")
plt.title("Scatter Plot - Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# ==============================
# Findings
# ==============================
"""
📌 Observations:
1. Sepal length shows variation across samples, trending upwards in the dataset.
2. Virginica species generally have the largest petal length.
3. Sepal width distribution is roughly normal, centered around 3 cm.
4. Strong positive correlation between sepal length and petal length, especially for virginica.
"""