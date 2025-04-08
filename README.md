# Email Spam Detection System

This project implements a machine learning system to classify emails as spam or ham (non-spam) using text processing and classification algorithms.

## Features

- **Text Preprocessing**:
  - Lowercasing
  - URL/HTML/punctuation removal
  - Number removal
  - NLTK tokenization
  - Stopword removal

- **Models Implemented**:
  - Logistic Regression (95.25% accuracy)
  - Decision Tree (93.99% accuracy)
  - Random Forest (97.31% accuracy - best performing)

- **Evaluation Metrics**:
  - Accuracy, precision, recall, F1-score
  - Classification reports
  - Confusion matrices
  - Feature importance analysis

## Key Findings

The Random Forest model performed best with 97.31% accuracy. Top important features for spam detection include:

1. free (0.0315)
2. claim (0.0303)  
3. call (0.0298)
4. mobile (0.0286)
5. txt (0.0248)

## Requirements

- Python 3.6+
- Required packages:
  ```
  pandas
  scikit-learn
  nltk
  matplotlib
  seaborn
  joblib
  ```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

Run the spam detection script:
```bash
python spam_detection.py
```

This will:
1. Preprocess the data
2. Train and evaluate models
3. Show performance metrics
4. Save the best model (Random Forest) to `spam_classifier.joblib`

## Saved Models

- `spam_classifier.joblib`: Trained Random Forest model
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer for text processing

## Dataset

The system uses the SMS Spam Collection Dataset from UCI Machine Learning Repository with columns:
- `label`: 'ham' or 'spam'
- `text`: Raw message content

## Future Improvements

- Experiment with deep learning models
- Add email header analysis
- Implement real-time classification API
- Add more sophisticated feature engineering
