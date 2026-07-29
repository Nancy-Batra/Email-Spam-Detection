# Email-Spam-Detection
This is a Machine Learning project that classifies emails as Spam or Not Spam. The system uses NLP techniques to preprocess email text, converts the text into numerical features using TF-IDF Vectorization, and trains machine learning models to perform classification.  Two algorithms were trained and compared: Naive Bayes and Logistic Regression.

The best-performing model is automatically selected, saved using Pickle and used in a Tkinter GUI for the email prediction.

## Features

- Email text preprocessing
- TF-IDF feature extraction
- Naive Bayes classifier
- Logistic Regression classifier
- Model Accuracy Comparison
- Save trained model using Pickle
- Tkinter GUI for prediction

## Dataset

The dataset is a CSV file containing email messages and their corresponding labels. Each email is classified into one of two categories:
- Spam (1)
- Not Spam (0)

**Example:**

| Text                                                              | Label    |
| ----------------------------------------------------------------- | -------- |
| "Congratulations! You've won a free iPhone. Click here to claim." | 1        |
| "Hi, can we reschedule our meeting to tomorrow?"                  | 0        |


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Tkinter
- Pickle
- 

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

## Results

### Naive Bayes
- Accuracy: **98.17%**

### Logistic Regression
- Accuracy: **98.43%**

Logistic Regression achieved the highest accuracy and was selected as the final model.

## GUI

The GUI allows users to:

- Enter an email
- Predict Spam or Not Spam
- Display prediction probabilities
- Clear the input


## Future Improvements

- Train on a larger email dataset
- Add support for multiple languages
- Deploy as a web application
- Improve prediction confidence visualization

