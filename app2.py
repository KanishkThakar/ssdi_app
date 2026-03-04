import os
import pandas as pd
import numpy as np
import io
import base64
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# --- Database Configuration ---
# Uses Render's DATABASE_URL or a local sqlite file for development
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///soul_yatri.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model for Soul Yatri Users
class SoulUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False) # e.g., 'Soul Seeker', 'Healing Partner'

# Create the database tables
with app.app_context():
    db.create_all()

# --- Helper Function for Machine Learning ---
def perform_classification(df, target, features, test_size, seed):
    # Data Cleaning
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns: df = df.drop(columns=["ID"])

    X = df[features].copy()
    y = df[target].copy()

    # Preprocessing
    X = X.fillna(X.mode().iloc[0])
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Modeling
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Plotting
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return acc, report, plot_url

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            df = pd.read_csv(file) if file else pd.read_csv("Credit.csv")
            
            target = request.form.get('target')
            features = request.form.getlist('features')
            test_size = float(request.form.get('test_size', 0.2))
            seed = int(request.form.get('seed', 42))

            acc, report, plot = perform_classification(df, target, features, test_size, seed)
            return jsonify({"accuracy": f"{acc:.4f}", "report": report, "plot": plot})
        except Exception as e:
            return jsonify({"error": str(e)})

    # Initial page load: Get columns from Credit.csv if it exists
    cols = pd.read_csv("Credit.csv").columns.tolist() if os.path.exists("Credit.csv") else []
    return render_template('index.html', columns=cols)

if __name__ == '__main__':
    app.run(debug=True)