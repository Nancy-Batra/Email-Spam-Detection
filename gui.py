import tkinter as tk
from tkinter import messagebox
import pickle
import string

# LOAD MODEL & VECTORIZER
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# preproccesing
def preprocess(text):

    text = text.lower()
    text = text.translate( str.maketrans("", "", string.punctuation) )
    return text

#pred func
def predict():
    email = textbox.get("1.0", tk.END).strip()

    if email == "":
        messagebox.showwarning(
            "Warning",
            "Please enter an email."
        )
        return

    processed = preprocess(email)
    email_vector = vectorizer.transform([processed])
    prediction = model.predict(email_vector)
    probability = model.predict_proba(email_vector)

    spam_probability = probability[0][1] * 100
    not_spam_probability = probability[0][0] * 100

    if prediction[0] == 1:
        result.config( text=f"""Prediction : SPAM
        Spam Probability : {spam_probability:.2f}%
        Not Spam Probability : {not_spam_probability:.2f}%""",
        fg="red"
        )
    else:
        result.config(
            text=f"""Prediction : NOT SPAM
            Not Spam Probability : {not_spam_probability:.2f}%
            Spam Probability : {spam_probability:.2f}%""",
            fg="green"
            )
        
# clear func
def clear():
    textbox.delete("1.0", tk.END)
    result.config(text="")

# gui

root = tk.Tk()
root.title("Email Spam Detection")
root.geometry("750x550")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Email Spam Detection",
    font=("Arial",20,"bold")
)
title.pack(pady=15)

instruction = tk.Label(
    root,
    text="Enter an Email Message",
    font=("Arial",13)
)
instruction.pack()

textbox = tk.Text(
    root,
    width=70,
    height=12,
    font=("Arial",11)
)

textbox.pack(pady=10)
predict_button = tk.Button(
    root,
    text="Predict",
    command=predict,
    font=("Arial",13),
    width=15
)

predict_button.pack(pady=8)

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear,
    font=("Arial",13),
    width=15
)
clear_button.pack()

result = tk.Label(
    root,
    text="",
    font=("Arial",16,"bold")
)
result.pack(pady=20)
root.mainloop()