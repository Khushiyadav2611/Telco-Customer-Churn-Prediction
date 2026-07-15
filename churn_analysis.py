# ============================================================
# Telco Customer Churn Prediction
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
df = pd.read_excel("telco_ibm_churn.xlsx")

print(df.head())
print(df.info())
print(df.shape)

# ------------------------------------------------------------
# 2. Clean Data
# ------------------------------------------------------------
# 'Total Charges' was stored as text due to 11 blank rows.
# All 11 belong to customers with Tenure Months = 0 (brand new,
# no billing history yet) -> fill with 0, not the column mean.
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'] = df['Total Charges'].fillna(0)

# 'Churn Reason' has ~5,174 blanks -> these are customers who
# never churned, so a reason legitimately doesn't apply. Left as-is.

# ------------------------------------------------------------
# 3. Exploratory Data Analysis
# ------------------------------------------------------------

# Top reasons customers churn
print(df['Churn Reason'].value_counts())

df['Churn Reason'].value_counts().plot(kind='barh')
plt.title('Reasons for Customer Churn')
plt.xlabel('Number of Customers')
plt.ylabel('Reason')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('churn_reasons_chart.png')
plt.show()

# Churn rate by Contract type
print(df.groupby('Contract')['Churn Value'].mean())

# Churn rate by Internet Service type
print(df.groupby('Internet Service')['Churn Value'].mean())

# Combined view: Contract x Internet Service
print(df.groupby(['Contract', 'Internet Service'])['Churn Value'].mean())

pivot = df.groupby(['Contract', 'Internet Service'])['Churn Value'].mean().unstack()
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Reds')
plt.title('Churn Rate by Contract & Internet Service')
plt.savefig('churn_heatmap.png')
plt.show()

# ------------------------------------------------------------
# 4. Prepare Features for Modeling
# ------------------------------------------------------------
features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Contract', 'Internet Service']
X = df[features]
y = df['Churn Value']

X = pd.get_dummies(X, columns=['Contract', 'Internet Service'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 5. Model 1 - Logistic Regression (default / baseline)
# ------------------------------------------------------------
log_reg_default = LogisticRegression(max_iter=1000)
log_reg_default.fit(X_train, y_train)

y_pred_default = log_reg_default.predict(X_test)

print("Logistic Regression (default) accuracy:", accuracy_score(y_test, y_pred_default))
print(confusion_matrix(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default))

# ------------------------------------------------------------
# 6. Model 2 - Logistic Regression (class_weight='balanced')
#    -> Final chosen model (best recall for churn class)
# ------------------------------------------------------------
log_reg_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_balanced.fit(X_train, y_train)

y_pred_balanced = log_reg_balanced.predict(X_test)

print("Logistic Regression (balanced):")
print(confusion_matrix(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced))

# ------------------------------------------------------------
# 7. Model 3 - Random Forest (comparison model)
# ------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest (balanced):")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Feature importance (used to cross-check EDA findings)
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

# ------------------------------------------------------------
# Final model: Logistic Regression (class_weight='balanced')
# Chosen for highest recall on the churn class (0.80), since
# missing an actual churner is costlier to the business than
# a false alarm.
# ------------------------------------------------------------
