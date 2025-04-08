import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
required_nltk = ['punkt', 'stopwords', 'punkt_tab']
for resource in required_nltk:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt_tab' else resource)
    except LookupError:
        nltk.download(resource)

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]  # Keep only label and text columns
    df.columns = ['label', 'text']
    return df

def preprocess_text(text):
    # Text cleaning
    text = text.lower()  # Lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    
    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

def main():
    # Load and prepare data
    df = load_data('dataset/spam.csv')
    df['text'] = df['text'].apply(preprocess_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    print("\nSplitting data into train/test sets (80%/20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize models with explicit configurations
    print("\nInitializing models with following configurations:")
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'config': 'max_iter=1000'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(max_depth=5),
            'config': 'max_depth=5'
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100),
            'config': 'n_estimators=100'
        }
    }
    
    # Print model configurations
    for name, model_info in models.items():
        print(f"- {name}: {model_info['config']}")
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = {}
    for name, model_info in models.items():
        model = model_info['model']
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classification Report:")
        print(metrics['report'])
    
    # Save best model (Random Forest) and show feature importance
    best_model = RandomForestClassifier(n_estimators=100)
    best_model.fit(X_train_vec, y_train)
    
    # Get and display top 10 important features
    importances = best_model.feature_importances_
    top_indices = importances.argsort()[-10:][::-1]  # Top 10 indices, highest first
    top_features = vectorizer.get_feature_names_out()[top_indices]
    top_importance = importances[top_indices]
    
    print("\nTop 10 Important Features for Spam Detection:")
    for feature, importance in zip(top_features, top_importance):
        print(f"{feature}: {importance:.4f}")
    
    # Save model and vectorizer
    import joblib
    joblib.dump(best_model, 'spam_classifier.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

if __name__ == '__main__':
    main()
