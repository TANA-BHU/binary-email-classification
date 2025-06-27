import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os
import joblib


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label'].values

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(vectorizer, "artifacts/vectorizer.pkl")


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer

def save_confusion_matrix(y_true, y_pred, model_name, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

def save_metrics_to_csv(results, save_path="outputs/model_metrics.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_df = pd.DataFrame(results)

    # Append if file exists
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["model"], keep='last')
    else:
        combined_df = new_df

    combined_df.to_csv(save_path, index=False)
    print(f"\nUpdated model metrics saved to {save_path}")

def plot_model_metrics_histogram(metrics_path="outputs/model_metrics.csv", save_path="outputs/metrics_histogram.png"):
    if not os.path.exists(metrics_path):
        print(f"No metrics file found at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    melted_df = df.melt(id_vars='model', var_name='metric', value_name='score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x='metric', y='score', hue='model')
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nHistogram of metrics saved to {save_path}")

def plot_auc_curve(all_probs, y_test, save_path="outputs/auc_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))

    for model_name, y_prob in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nAUC curve saved to {save_path}")
