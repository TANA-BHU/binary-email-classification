# models/trainers.py
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Models.dnn import DNNModel
from utils.processing import save_confusion_matrix

def summarize_results(name, y_test, y_pred):
    print(f"\n{name} Performance")
    print(classification_report(y_test, y_pred))
    save_confusion_matrix(y_test, y_pred, model_name=name)
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

def run_lr(X_train, X_test, y_train, y_test, return_probs=False):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = summarize_results("Logistic Regression", y_test, y_pred)
    y_prob = model.predict_proba(X_test)[:, 1] if return_probs else None
    return {"metrics": metrics, "probs": y_prob, "model": model} if return_probs else metrics

def run_xgb(X_train, X_test, y_train, y_test, return_probs=False):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = summarize_results("XGBoost", y_test, y_pred)
    y_prob = model.predict_proba(X_test)[:, 1] if return_probs else None
    return {"metrics": metrics, "probs": y_prob, "model": model} if return_probs else metrics

def run_dnn(X_train, X_test, y_train, y_test, input_dim, return_probs=False, epochs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNModel(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_tensor.to(device)).cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
        metrics = summarize_results("DNN", y_test, y_pred)
    # return {"metrics": metrics, "probs": y_prob.flatten()} if return_probs else metrics
    return {"metrics": metrics, "probs": y_prob.flatten(), "model": model} if return_probs else metrics