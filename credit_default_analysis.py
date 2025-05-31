# Purpose: This script performs a comprehensive analysis to predict credit card defaults
# using a dataset of client financial behaviors. It includes data cleaning, exploratory
# data analysis, feature engineering, and model evaluation with detailed visualizations.

# Stage 1: Import Required Libraries
# Importing libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.ticker as ticker

# Stage 2: Load and Validate Dataset
# Loading the dataset and handling potential errors
try:
    dataset = pd.read_excel('default of credit card clients.xlsx', header=1)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Stage 3: Data Cleaning and Preprocessing
# Renaming columns for better interpretability
dataset = dataset.rename(columns={
    'LIMIT_BAL': 'Credit_Limit',
    'SEX': 'Gender',
    'EDUCATION': 'Education_Level',
    'MARRIAGE': 'Marital_Status',
    'AGE': 'Client_Age',
    'PAY_0': 'Repay_Sep_2005',
    'PAY_2': 'Repay_Aug_2005',
    'PAY_3': 'Repay_Jul_2005',
    'PAY_4': 'Repay_Jun_2005',
    'PAY_5': 'Repay_May_2005',
    'PAY_6': 'Repay_Apr_2005',
    'BILL_AMT1': 'Bill_Sep_2005',
    'BILL_AMT2': 'Bill_Aug_2005',
    'BILL_AMT3': 'Bill_Jul_2005',
    'BILL_AMT4': 'Bill_Jun_2005',
    'BILL_AMT5': 'Bill_May_2005',
    'BILL_AMT6': 'Bill_Apr_2005',
    'PAY_AMT1': 'Payment_Sep_2005',
    'PAY_AMT2': 'Payment_Aug_2005',
    'PAY_AMT3': 'Payment_Jul_2005',
    'PAY_AMT4': 'Payment_Jun_2005',
    'PAY_AMT5': 'Payment_May_2005',
    'PAY_AMT6': 'Payment_Apr_2005',
    'default payment next month': 'Default_Status'
})

# Checking dataset structure and missing values
print("\nDataset Overview:")
print(dataset.info())
print("\nMissing Values Check:")
print(dataset.isnull().sum())
print("\nDefault Status Distribution:")
print(dataset['Default_Status'].value_counts(normalize=True))

# Cleaning Education_Level: Mapping invalid values (0, 5, 6) to 'Other' (4)
dataset['Education_Level'] = dataset['Education_Level'].replace([0, 5, 6], 4)
print("\nEducation Level Distribution (Post-Cleaning):")
print(dataset['Education_Level'].value_counts().sort_index())

# Cleaning Marital_Status: Mapping invalid value (0) to 'Other' (3)
dataset['Marital_Status'] = dataset['Marital_Status'].replace(0, 3)
print("\nMarital Status Distribution (Post-Cleaning):")
print(dataset['Marital_Status'].value_counts().sort_index())

# Stage 4: Exploratory Data Analysis (EDA)
# Visualization 1: Default Status Distribution
# Purpose: Visualize the imbalance in the target variable
plt.figure(figsize=(8, 5))
sns.countplot(x='Default_Status', data=dataset, palette='Set1')
plt.title('Distribution of Default Status')
plt.xlabel('Default Status (0 = No, 1 = Yes)')
plt.ylabel('Number of Clients')
plt.xticks([0, 1], ['Non-Default', 'Default'])
plt.show()
# Insight: The plot reveals a class imbalance (~78% non-defaulters vs. ~22% defaulters),
# necessitating techniques like SMOTE to balance the dataset for better model performance.

# Visualization 2: Feature Correlation Heatmap
# Purpose: Identify relationships between features
plt.figure(figsize=(14, 10))
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix, cmap='viridis', annot=False, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.show()
# Insight: Strong correlations among bill amounts across months suggest multicollinearity,
# while repayment status variables show moderate correlation with default status.

# Visualization 3: Credit Limit Distribution by Default Status
# Purpose: Analyze how credit limits differ between defaulters and non-defaulters
plt.figure(figsize=(10, 6))
sns.histplot(data=dataset, x='Credit_Limit', hue='Default_Status', bins=30, multiple='stack', palette='Set2')
plt.title('Credit Limit Distribution by Default Status')
plt.xlabel('Credit Limit (NT Dollar)')
plt.ylabel('Count')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
plt.legend(title='Default Status', labels=['Non-Default', 'Default'])
plt.show()
# Insight: Defaulters tend to have lower credit limits (<200K NT$), indicating credit limit
# as a potential predictor of default risk.

# Visualization 4: Education Level vs. Default Status
# Purpose: Explore default rates across education levels
plt.figure(figsize=(10, 6))
sns.countplot(x='Education_Level', hue='Default_Status', data=dataset, palette='Set3')
plt.title('Education Level vs. Default Status')
plt.xlabel('Education Level (1=Grad School, 2=University, 3=High School, 4=Other)')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3], ['Grad School', 'University', 'High School', 'Other'])
plt.legend(title='Default Status', labels=['Non-Default', 'Default'])
plt.show()
# Insight: Higher default proportions in High School and Other categories suggest
# education level as a risk factor.

# Visualization 5: Default Probability by Age Group
# Purpose: Assess how age impacts default likelihood
dataset['Age_Group'] = pd.cut(dataset['Client_Age'], bins=[20, 30, 40, 50, 60, 70, 80],
                             labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
age_default = dataset.groupby('Age_Group', observed=False)['Default_Status'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Age_Group', y='Default_Status', hue='Age_Group', data=age_default, palette='Set2', legend=False)
plt.title('Default Probability by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Default Probability')
plt.ylim(0, 1)
plt.show()
dataset.drop('Age_Group', axis=1, inplace=True)
# Insight: Younger clients (20-30) and older clients (>50) show slightly higher default rates,
# suggesting age-based risk profiling.

# Stage 5: Feature Engineering
# Creating new features to capture financial behavior
dataset['Pay_to_Bill_Ratio'] = dataset['Payment_Sep_2005'] / dataset['Bill_Sep_2005']
dataset['Pay_to_Bill_Ratio'] = dataset['Pay_to_Bill_Ratio'].clip(lower=0, upper=10).replace([np.inf, -np.inf], np.nan).fillna(0)
dataset['Credit_Usage_Ratio'] = dataset['Bill_Sep_2005'] / dataset['Credit_Limit']

# Visualization 6: Pay-to-Bill Ratio Density
# Purpose: Compare payment behavior between defaulters and non-defaulters
plt.figure(figsize=(10, 6))
sns.kdeplot(data=dataset, x='Pay_to_Bill_Ratio', hue='Default_Status', palette='Set2', common_norm=False)
plt.title('Pay-to-Bill Ratio Density by Default Status')
plt.xlabel('Pay-to-Bill Ratio (Sep 2005)')
plt.ylabel('Density')
plt.xlim(0, 5)
plt.legend(title='Default Status', labels=['Non-Default', 'Default'])
plt.show()
# Insight: Defaulters cluster at lower ratios (<0.5), indicating poor payment behavior as a strong default predictor.

# Visualization 7: Credit Usage Ratio Density
# Purpose: Examine credit utilization patterns
plt.figure(figsize=(10, 6))
sns.kdeplot(data=dataset, x='Credit_Usage_Ratio', hue='Default_Status', palette='Set2', common_norm=False)
plt.title('Credit Usage Ratio Density by Default Status')
plt.xlabel('Credit Usage Ratio (Sep 2005)')
plt.ylabel('Density')
plt.xlim(-0.5, 2)
plt.legend(title='Default Status', labels=['Non-Default', 'Default'])
plt.show()
# Insight: Higher credit usage ratios correlate with increased default risk.

# Visualization 8: Average Default Rates by Repayment Status
# Purpose: Analyze repayment status impact on default probability
repay_cols = ['Repay_Sep_2005', 'Repay_Aug_2005', 'Repay_Jul_2005',
              'Repay_Jun_2005', 'Repay_May_2005', 'Repay_Apr_2005']
for col in repay_cols:
    dataset[col] = dataset[col].replace(-2, -1).clip(lower=-1, upper=9)
default_rates = []
for col in repay_cols:
    rates = dataset.groupby(col)['Default_Status'].mean().reset_index()
    rates = rates.rename(columns={col: 'Repay_Status'})
    default_rates.append(rates)
avg_rates = pd.concat(default_rates).groupby('Repay_Status')['Default_Status'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Repay_Status', y='Default_Status', hue='Repay_Status', data=avg_rates, palette='Blues', legend=False)
plt.title('Average Default Rates by Repayment Status')
plt.xlabel('Repayment Status (-1=Paid, 0=Revolving, 1-9=Months Delayed)')
plt.ylabel('Default Rate')
plt.ylim(0, 1)
plt.show()
# Insight: Higher repayment delays (e.g., 1-9 months) strongly correlate with default risk.

# Stage 6: Data Preparation for Modeling
# Encoding categorical variables and preparing features
dataset = pd.get_dummies(dataset, columns=['Gender', 'Education_Level', 'Marital_Status'], drop_first=True)
X = dataset.drop(['ID', 'Default_Status'], axis=1)
y = dataset['Default_Status']
print("\nClass Distribution Before SMOTE:", Counter(y))

# Balancing classes using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print("Class Distribution After SMOTE:", Counter(y_balanced))

# Visualization 9: Class Distribution Before and After SMOTE
# Purpose: Visualize the effect of SMOTE on class balance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y, hue=y, palette='Set1', legend=False)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Default Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Default', 'Default'])
plt.subplot(1, 2, 2)
sns.countplot(x=y_balanced, hue=y_balanced, palette='Set1', legend=False)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Default Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Default', 'Default'])
plt.tight_layout()
plt.show()
# Insight: SMOTE balances the classes, ensuring equal representation of defaulters and non-defaulters.

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Scaling features for models sensitive to feature scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Stage 7: Model Training and Evaluation
# Defining models for comparison
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000),
    'Decision_Tree': DecisionTreeClassifier(),
    'Random_Forest': RandomForestClassifier(),
    'Gradient_Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'Naive_Bayes': GaussianNB()
}

# Training and evaluating models
results = []
model_probabilities = {}
for model_name, model in models.items():
    # Fitting model with appropriate data (scaled for KNN and Logistic Regression)
    if model_name in ['KNN', 'Logistic_Regression']:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(y_test))
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(y_test))
    
    model_probabilities[model_name] = probabilities
    cm = confusion_matrix(y_test, predictions)
    error_rate = (cm[0, 1] + cm[1, 0]) / len(y_test)
    
    # Storing performance metrics
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Error_Rate': error_rate,
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1_Score': f1_score(y_test, predictions),
        'ROC_AUC': roc_auc_score(y_test, probabilities) if probabilities.sum() > 0 else 0
    })
    
    # Visualization 10: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Insight: High true positives and low false negatives indicate effective default prediction.

# Stage 8: Model Performance Summary
# Displaying results in a table
results_df = pd.DataFrame(results)
print("\nModel Performance Metrics:")
print(results_df.round(4))

# Visualization 11: ROC Curves
# Purpose: Compare model discrimination ability
plt.figure(figsize=(10, 8))
for model_name, probs in model_probabilities.items():
    if probs.sum() > 0:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
# Insight: Random Forest typically shows the highest AUC, indicating superior performance.

# Visualization 12: Precision-Recall Curves
# Purpose: Evaluate model performance on imbalanced positive class
plt.figure(figsize=(10, 8))
for model_name, probs in model_probabilities.items():
    if probs.sum() > 0:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.plot(recall, precision, label=model_name)
plt.title('Precision-Recall Curves for All Models')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()
# Insight: High precision and recall for Random Forest make it suitable for default prediction.

# Visualization 13: Feature Importance (Random Forest)
# Purpose: Identify key predictors
rf_model = models['Random_Forest']
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_imp.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()
# Insight: Repayment status and credit limit are top predictors, guiding risk assessment strategies.

# Stage 9: Export Results
# Saving model performance to CSV
results_df.to_csv('credit_default_model_results.csv', index=False)
print("\nResults exported to 'credit_default_model_results.csv'")