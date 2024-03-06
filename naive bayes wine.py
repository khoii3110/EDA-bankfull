import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
data = pd.read_csv("winequality-red.csv", encoding="utf-8")

#print(data.columns);
#data.columns =  data.columns.str.strip();
#print(data.head());

# 2a. Print information about the attribute columns
column_info = {
    "fixed acidity": "Fixed acidity",
    "volatile acidity": "Volatile acidity",
    "citric acid": "Citric acid",
    "residual sugar": "Residual sugar",
    "chlorides": "Chlorides",
    "free sulfur dioxide": "Free sulfur dioxide",
    "total sulfur dioxide": "Total sulfur dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol",
    "quality": "Quality"
}
print("Information about attribute columns:")
for col, info in column_info.items():
    print(f"{col}: {info}")

# 2b. Find the correlation between the "alcohol" attribute and wine quality
sns.scatterplot(x='alcohol', y='quality', data=data)
plt.title('Correlation between alcohol content and wine quality')
plt.xlabel('Alcohol content')
plt.ylabel('Wine quality')
plt.show()

# 2c. Find the correlation between all attributes and wine quality
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between attributes and wine quality')
plt.show()

# 2d. Remove irrelevant attributes and build a linear correlation between the 3 most important attributes and wine quality
important_attributes = ['alcohol', 'volatile acidity', 'sulphates', 'quality']
important_data = data[important_attributes]

# Remove missing values
important_data.dropna(inplace=True)

# Build linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = important_data.drop('quality', axis=1)
y = important_data['quality']

model.fit(X, y)


# Print coefficients and intercept
print("Coefficients of the linear regression model:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")


