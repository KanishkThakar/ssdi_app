import os
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as t_dist

app = Flask(__name__)

# --- Database Configuration ---
# Render provides DATABASE_URL with postgres:// prefix; SQLAlchemy needs postgresql://
database_url = os.getenv('DATABASE_URL', 'sqlite:///ssdi_app.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

db = SQLAlchemy(app)


# ===========================
# Database Models
# ===========================
class ClassificationResult(db.Model):
    """Stores each ML classification run."""
    __tablename__ = 'classification_results'
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(255), nullable=False, default='Uploaded CSV')
    target_column = db.Column(db.String(100), nullable=False)
    features_used = db.Column(db.Text, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    report = db.Column(db.Text, nullable=False)
    test_size = db.Column(db.Float, default=0.2)
    random_seed = db.Column(db.Integer, default=42)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class TTestResult(db.Model):
    """Stores each T-Test calculation."""
    __tablename__ = 'ttest_results'
    id = db.Column(db.Integer, primary_key=True)
    sample_data = db.Column(db.Text, nullable=False)
    null_mean = db.Column(db.Float, nullable=False)
    alpha = db.Column(db.Float, nullable=False)
    alternative = db.Column(db.String(20), nullable=False)
    sample_mean = db.Column(db.Float, nullable=False)
    sample_std = db.Column(db.Float, nullable=False)
    t_statistic = db.Column(db.Float, nullable=False)
    degrees_of_freedom = db.Column(db.Integer, nullable=False)
    p_value = db.Column(db.Float, nullable=False)
    reject_null = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# Create all tables
with app.app_context():
    db.create_all()


# ===========================
# Helper Functions
# ===========================
def perform_classification(df, target, features, test_size, seed):
    """Run Naive Bayes classification and return results."""
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df[features].copy()
    y = df[target].copy()

    # Preprocessing
    X = X.fillna(X.mode().iloc[0])
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Generate confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    plt.style.use('dark_background')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="magma",
                linewidths=0.5, linecolor='#2a2a4a',
                cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix', fontsize=14, color='#e0e0ff', pad=15)
    plt.xlabel('Predicted', fontsize=12, color='#b0b0d0')
    plt.ylabel('Actual', fontsize=12, color='#b0b0d0')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return acc, report, plot_url


def perform_ttest(data, mu0, alpha=0.05, alternative="two-sided"):
    """Perform a one-sample T-Test."""
    data = np.array(data)
    n = len(data)
    xbar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    t_cal = (xbar - mu0) / se
    df = n - 1

    if alternative == "two-sided":
        p_value = 2 * (1 - t_dist.cdf(abs(t_cal), df))
    elif alternative == "greater":
        p_value = 1 - t_dist.cdf(t_cal, df)
    elif alternative == "less":
        p_value = t_dist.cdf(t_cal, df)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    reject = p_value < alpha
    return xbar, s, t_cal, df, p_value, reject


# ===========================
# Routes
# ===========================
@app.route('/')
def index():
    """Main page."""
    cols = []
    if os.path.exists("Credit.csv"):
        try:
            cols = pd.read_csv("Credit.csv").columns.tolist()
        except Exception:
            pass
    return render_template('index.html', columns=cols)


@app.route('/classify', methods=['POST'])
def classify():
    """Run Naive Bayes classification."""
    try:
        file = request.files.get('file')
        if file and file.filename:
            df = pd.read_csv(file)
            dataset_name = file.filename
        elif os.path.exists("Credit.csv"):
            df = pd.read_csv("Credit.csv")
            dataset_name = "Credit.csv"
        else:
            return jsonify({"error": "No file uploaded and no default dataset found."}), 400

        target = request.form.get('target')
        features = request.form.getlist('features')
        test_size = float(request.form.get('test_size', 0.2))
        seed = int(request.form.get('seed', 42))

        if not target:
            return jsonify({"error": "Please select a target column."}), 400
        if not features:
            return jsonify({"error": "Please select at least one feature."}), 400

        acc, report, plot = perform_classification(df, target, features, test_size, seed)

        # Save to database
        result = ClassificationResult(
            dataset_name=dataset_name,
            target_column=target,
            features_used=', '.join(features),
            accuracy=acc,
            report=report,
            test_size=test_size,
            random_seed=seed
        )
        db.session.add(result)
        db.session.commit()

        return jsonify({
            "id": result.id,
            "accuracy": f"{acc:.4f}",
            "report": report,
            "plot": plot,
            "dataset": dataset_name,
            "timestamp": result.created_at.strftime('%Y-%m-%d %H:%M')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ttest', methods=['POST'])
def ttest():
    """Run a one-sample T-Test."""
    try:
        data_input = request.form.get('data', '')
        mu0 = float(request.form.get('mu0', 0))
        alpha = float(request.form.get('alpha', 0.05))
        alternative = request.form.get('alternative', 'two-sided')

        data = [float(x.strip()) for x in data_input.split(',') if x.strip()]
        if len(data) < 2:
            return jsonify({"error": "Please enter at least 2 data values."}), 400

        xbar, s, t_cal, df, p_value, reject = perform_ttest(data, mu0, alpha, alternative)

        # Cast numpy types to native Python types for JSON serialization
        xbar = float(xbar)
        s = float(s)
        t_cal = float(t_cal)
        df = int(df)
        p_value = float(p_value)
        reject = bool(reject)

        # Save to database
        result = TTestResult(
            sample_data=data_input,
            null_mean=mu0,
            alpha=alpha,
            alternative=alternative,
            sample_mean=xbar,
            sample_std=s,
            t_statistic=t_cal,
            degrees_of_freedom=df,
            p_value=p_value,
            reject_null=reject
        )
        db.session.add(result)
        db.session.commit()

        return jsonify({
            "id": result.id,
            "sample_mean": f"{xbar:.4f}",
            "sample_std": f"{s:.4f}",
            "t_statistic": f"{t_cal:.4f}",
            "degrees_of_freedom": df,
            "p_value": f"{p_value:.6f}",
            "reject": reject,
            "decision": "Reject H₀" if reject else "Fail to Reject H₀",
            "timestamp": result.created_at.strftime('%Y-%m-%d %H:%M')
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history')
def history():
    """Get classification and T-Test history."""
    try:
        classifications = ClassificationResult.query.order_by(
            ClassificationResult.created_at.desc()
        ).limit(20).all()

        ttests = TTestResult.query.order_by(
            TTestResult.created_at.desc()
        ).limit(20).all()

        return jsonify({
            "classifications": [{
                "id": r.id,
                "dataset": r.dataset_name,
                "target": r.target_column,
                "accuracy": f"{r.accuracy:.4f}",
                "test_size": r.test_size,
                "date": r.created_at.strftime('%Y-%m-%d %H:%M')
            } for r in classifications],
            "ttests": [{
                "id": r.id,
                "null_mean": r.null_mean,
                "t_statistic": f"{r.t_statistic:.4f}",
                "p_value": f"{r.p_value:.6f}",
                "decision": "Reject H₀" if r.reject_null else "Fail to Reject H₀",
                "date": r.created_at.strftime('%Y-%m-%d %H:%M')
            } for r in ttests]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history/<string:type>/<int:id>', methods=['DELETE'])
def delete_history(type, id):
    """Delete a history record."""
    try:
        if type == 'classification':
            record = ClassificationResult.query.get_or_404(id)
        elif type == 'ttest':
            record = TTestResult.query.get_or_404(id)
        else:
            return jsonify({"error": "Invalid type"}), 400

        db.session.delete(record)
        db.session.commit()
        return jsonify({"message": "Deleted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/columns', methods=['POST'])
def get_columns():
    """Get column names from an uploaded CSV file."""
    try:
        file = request.files.get('file')
        if file and file.filename:
            df = pd.read_csv(file)
            return jsonify({"columns": df.columns.tolist()})
        return jsonify({"columns": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
