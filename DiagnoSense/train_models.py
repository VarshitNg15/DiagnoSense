# Import required libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import shap

# Load and preprocess the data
print("Loading and preprocessing data...")
data = pd.read_csv("Training.csv").dropna(axis=1)

# Prepare features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save encoder classes
np.save('encoder_classes.npy', encoder.classes_)

# Feature scaling using Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Dimensionality reduction using PCA
pca = PCA(n_components=50)  # Reduce to 50 components
X_pca = pca.fit_transform(X_scaled)
joblib.dump(pca, 'pca_model.pkl')

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Define models
print("Training SVM model...")
svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'final_svm_model.pkl')
print("SVM model trained and saved.")

print("Training Naive Bayes model...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
joblib.dump(nb_model, 'final_nb_model.pkl')
print("Naive Bayes model trained and saved.")

print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'final_rf_model.pkl')
print("Random Forest model trained and saved.")

# Model stacking for better accuracy
estimators = [('svm', svm_model), ('nb', nb_model), ('rf', rf_model)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))
stack_model.fit(X_train, y_train)
joblib.dump(stack_model, 'final_stack_model.pkl')
print("Stacking model trained and saved.")

# Model evaluation
print("\nEvaluating models...")
models = {'SVM': svm_model, 'Naive Bayes': nb_model, 'Random Forest': rf_model, 'Stacking Model': stack_model}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    print(f"\n{name} Model Performance:")
    print(classification_report(y_test, y_pred))
    if y_prob is not None:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob, multi_class='ovr')}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Explainable AI using SHAP for feature importance
print("\nGenerating SHAP explanations for Random Forest model...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=data.columns[:-1])
