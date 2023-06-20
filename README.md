
# Mental Health Fitness Tracker ⚠️Readme Needs Update
The Mental Health Fitness Tracker project focuses on analyzing and predicting mental fitness levels of individuals from various countries with different mental disorders. It utilizes regression techniques to provide insights into mental health and make predictions based on the available data.


## INSTALLATION

To use the code and run the examples, follow these steps:

1. Ensure that you have Python 3.x installed on your system.
2. Install the required libraries by running the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
    
3. Download the project files and navigate to the project directory.
## USAGE

1. Import the necessary libraries:

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
```

2. Load and preprocess the dataset:

```bash
# Load the dataset using pandas
dataset = pd.read_csv('mental_health_data.csv')

# Preprocess the dataset (e.g., handle missing values, feature scaling)
# ...

# Split the dataset into input features (X) and target variable (y)
X = dataset.drop(columns=['mental_fitness'])
y = dataset['mental_fitness']
```

3. Split the dataset into training and testing sets:

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Fit and predict using the regression models:

```bash
# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)

# Elastic Net Regression
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_model.fit(X_train, y_train)
elastic_y_pred = elastic_model.predict(X_test)
```

5. Evaluate the model performance:

```bash
# Calculate the Mean Squared Error (MSE) and R-squared score for each model
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)

elastic_mse = mean_squared_error(y_test, elastic_y_pred)
elastic_r2 = r2_score(y_test, elastic_y_pred)
```

6. Visualize and compare the model performance:

```bash
# Create a bar plot for MSE scores
models = ['Ridge Regression', 'Elastic Net Regression']
mse_scores = [ridge_mse, elastic_mse]

plt.figure(figsize=(8, 4))
plt.bar(models, mse_scores)
plt.xlabel('Regression Model')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of MSE Scores')
plt.show()

# Create a bar plot for R-squared scores
r2_scores = [ridge_r2, elastic_r2]

plt.figure(figsize=(8, 4))
plt.bar(models, r2_scores)
plt.xlabel('Regression Model')
plt.ylabel('R-squared Score')
plt.title('Comparison of R-squared Scores')
plt.show()
```






## REFRENCES
- Datasets that were user in here were taken from [ourworldindia.org](https://ourworldindata.org/grapher/mental-and-substance-use-as-share-of-disease)

- This project was made during my internship period for [Edunet Foundation](https://edunetfoundation.org) in association with [IBM SkillsBuild](https://skillsbuild.org) and [AICTE](https://internship.aicte-india.org)