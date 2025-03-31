import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DiseasePredictor:
    def __init__(self):
        try:
            # Load the trained models
            self.stack_model = joblib.load('final_stack_model.pkl')
            self.rf_model = joblib.load('final_rf_model.pkl')
            self.nb_model = joblib.load('final_nb_model.pkl')
            self.svm_model = joblib.load('final_svm_model.pkl')
            
            # Load preprocessing components
            self.scaler = joblib.load('scaler.pkl')
            self.pca = joblib.load('pca_model.pkl')
            
            # Load symptom list from training data
            self.symptoms_df = pd.read_csv('Training.csv')
            
            # Get the expected number of features from the scaler
            self.expected_features = len(self.scaler.get_feature_names_out())
            
            # Ensure we have the correct number of features
            self.symptoms = self.symptoms_df.columns[:-1].tolist()[:self.expected_features]
            
            # Create label encoder from training data
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.symptoms_df['prognosis'])
            
            # Initialize stopwords
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_symptoms(self, text):
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Create a binary vector for symptoms with the correct number of features
        symptom_vector = np.zeros(self.expected_features)
        
        # Check for each symptom in the processed text
        for i, symptom in enumerate(self.symptoms):
            if symptom.lower() in processed_text:
                symptom_vector[i] = 1
        
        return symptom_vector
    
    def predict(self, symptoms_text):
        try:
            # Extract symptoms from text
            symptom_vector = self.extract_symptoms(symptoms_text)
            
            # Reshape for prediction
            X = symptom_vector.reshape(1, -1)
            
            # Create DataFrame with feature names to avoid warning
            X_df = pd.DataFrame(X, columns=self.scaler.get_feature_names_out())
            
            # Scale the features
            X_scaled = self.scaler.transform(X_df)
            
            # Apply PCA
            X_pca = self.pca.transform(X_scaled)
            
            # Get predictions from all models
            stack_pred = self.stack_model.predict(X_pca)
            rf_pred = self.rf_model.predict(X_pca)
            nb_pred = self.nb_model.predict(X_pca)
            svm_pred = self.svm_model.predict(X_pca)
            
            # Get probabilities for confidence calculation
            stack_prob = np.max(self.stack_model.predict_proba(X_pca))
            rf_prob = np.max(self.rf_model.predict_proba(X_pca))
            nb_prob = np.max(self.nb_model.predict_proba(X_pca))
            svm_prob = np.max(self.svm_model.predict_proba(X_pca))
            
            # Calculate average confidence
            confidence = np.mean([stack_prob, rf_prob, nb_prob, svm_prob]) * 100
            
            # Use majority voting for final prediction
            predictions = [stack_pred[0], rf_pred[0], nb_pred[0], svm_pred[0]]
            final_prediction = max(set(predictions), key=predictions.count)
            
            # Convert prediction to disease name
            disease = self.label_encoder.inverse_transform([final_prediction])[0]
            
            return {
                'disease': disease,
                'symptoms': symptoms_text,
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'disease': 'Error in prediction',
                'symptoms': symptoms_text,
                'confidence': 0,
                'error': str(e)
            } 