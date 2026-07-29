import os
import pickle
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay

os.makedirs("results", exist_ok=True)

df = pd.read_csv("emails_dataset.csv")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nClass Distribution")
print(df["spam"].value_counts())

original_text = df["text"].copy()

df["text"] = df["text"].fillna("")
df["text"] = df["text"].str.lower()

def remove_punctuation(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )

df["text"] = df["text"].apply(remove_punctuation)
stop_words = set(stopwords.words("english"))

print("\nOriginal Email\n")
print(original_text.iloc[0])

print("\nProcessed Email\n")
print(df["text"].iloc[0])

vectorizer = TfidfVectorizer( max_features=10000, ngram_range=(1,2),stop_words="english", lowercase=True )

X = vectorizer.fit_transform(df["text"])
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y )

# NAIVE BAYES
print("NAIVE BAYES MODEL")

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

print(f"Accuracy : {nb_accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred_nb))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_nb,
    cmap="Blues"
)

plt.title("Naive Bayes Confusion Matrix")
plt.savefig("results/naive_bayes_confusion_matrix.png")
plt.show()


# Logistic regression model
print("Logistic Rgression MODEL")

lr_classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_classifier.fit(X_train,y_train)
y_pred_lr = lr_classifier.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
)

print(f"Accuracy : {lr_accuracy*100:.2f}%")
print("\nClassification Report\n")
print(classification_report( y_test, y_pred_lr))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr,
    cmap="Greens"
)

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("results/logistic_regression_confusion_matrix.png")
plt.show()

#model comparison
print("MODEL COMPARISON")
print(f"Naive Bayes Accuracy        : {nb_accuracy*100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%")

plt.figure(figsize=(6,5))

models = [ "Naive Bayes", "Logistic Regression"]
accuracies = [ nb_accuracy*100, lr_accuracy*100 ]

plt.bar(models, accuracies)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.savefig("results/model_comparison.png")
plt.show()

# Saving best model
if lr_accuracy > nb_accuracy:
    best_model = lr_classifier
    best_prediction = y_pred_lr
    best_name = "Logistic Regression"
else:
    best_model = nb_classifier
    best_prediction = y_pred_nb
    best_name = "Naive Bayes"

print("\nBest Model :", best_name)

# saving model using PICKLE
import pickle

with open("spam_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel Saved Successfully!")


report = classification_report( y_test, best_prediction)

with open("results/classification_report.txt", "w") as file:
    file.write(report)

with open("results/model_results.txt", "w") as file:
    file.write("EMAIL SPAM DETECTION RESULTS\n\n")
    file.write(f"Naive Bayes Accuracy : {nb_accuracy*100:.2f}%\n")
    file.write(f"Logistic Regression Accuracy : {lr_accuracy*100:.2f}%\n")
    file.write(f"Best Model : {best_name}\n")

print("Results Saved Successfully!")
