# Email-Spam-Detection
This is a Machine Learning project that classifies emails as Spam or Not Spam. The system uses NLP techniques to preprocess email text, converts the text into numerical features using TF-IDF Vectorization, and trains machine learning models to perform classification.  Two algorithms were trained and compared: Naive Bayes and Logistic Regression.

The best-performing model is automatically selected, saved using Pickle and used in a Tkinter GUI for the email prediction.

---

## Features

- Email text preprocessing
- TF-IDF feature extraction
- Naive Bayes classifier
- Logistic Regression classifier
- Automatic model comparison
- Classification Report
- Confusion Matrix
- Model Accuracy Comparison
- Save trained model using Pickle
- Tkinter GUI for prediction

---

## Dataset

The dataset contains email messages labeled as:

- Spam
- Not Spam

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Tkinter
- Pickle

---

## Project Structure

```
Email-Spam-Detection/
│
├── main.py
├── main.ipynb
├── gui.py
├── emails_dataset.csv
├── spam_model.pkl
├── tfidf_vectorizer.pkl
│
├── results/
│   ├── naive_bayes_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── model_comparison.png
│   ├── classification_report.txt
│   └── model_results.txt
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Workflow

1. Load the dataset
2. Clean the email text
3. Convert text to lowercase
4. Remove punctuation
5. Extract TF-IDF features
6. Split the dataset into training and testing sets
7. Train Naive Bayes and Logistic Regression models
8. Compare model performance
9. Save the best model using Pickle
10. Predict emails using the Tkinter GUI

---

## Results

### Naive Bayes

- Accuracy: **98.17%**

### Logistic Regression

- Accuracy: **98.43%**

Logistic Regression achieved the highest accuracy and was selected as the final model.

---

## GUI

The GUI allows users to:

- Enter an email
- Predict Spam or Not Spam
- Display prediction probabilities
- Clear the input

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python main.py
```

### Launch the GUI

```bash
python gui.py
```

---

## Future Improvements

- Train on a larger email dataset
- Add support for multiple languages
- Deploy as a web application
- Improve prediction confidence visualization

---

## Author

Nancy Batra
