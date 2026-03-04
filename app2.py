import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

def process_and_train(df, target_column, feature_columns, test_size, random_state):
    # Remove common ID columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values
    X = X.fillna(X.mode().iloc[0])

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target
    if y.dtype == "object":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return accuracy, report, plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Handle File Upload
            file = request.files.get('file')
            if file:
                df = pd.read_csv(file)
            else:
                df = pd.read_csv("Credit.csv")

            target = request.form.get('target')
            features = request.form.getlist('features')
            test_size = float(request.form.get('test_size', 0.2))
            seed = int(request.form.get('seed', 42))

            if not target or not features:
                return jsonify({"error": "Please select target and features"})

            acc, report, plot = process_and_train(df, target, features, test_size, seed)
            
            return jsonify({
                "accuracy": f"{acc:.4f}",
                "report": report,
                "plot": plot
            })
        except Exception as e:
            return jsonify({"error": str(e)})

    # Initial Load
    cols = []
    if os.path.exists("Credit.csv"):
        cols = pd.read_csv("Credit.csv").columns.tolist()
    return render_template('index.html', columns=cols)

if __name__ == '__main__':
    app.run(debug=True)