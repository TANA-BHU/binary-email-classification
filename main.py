# Directory: phishing_detector

# main.py
from utils.processing import load_and_preprocess, save_metrics_to_csv, plot_model_metrics_histogram, plot_auc_curve
from Models.trainers import run_lr, run_xgb, run_dnn, DNNModel
import argparse
import pandas as pd
import joblib
import torch
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lr', 'xgb', 'dnn', 'all'], required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for DNN training")
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess(args.data)
    joblib.dump(vectorizer, "artifacts/vectorizer.pkl")

    results = []
    all_probs = {}

    if args.model == 'lr':
        result = run_lr(X_train, X_test, y_train, y_test, return_probs=True)
        results.append(result['metrics'])
        all_probs['Logistic Regression'] = result['probs']
        joblib.dump(result['model'], "artifacts/lr_model.pkl")

    elif args.model == 'xgb':
        result = run_xgb(X_train, X_test, y_train, y_test, return_probs=True)
        results.append(result['metrics'])
        all_probs['XGBoost'] = result['probs']
        joblib.dump(result['model'], "artifacts/xgb_model.pkl")

    elif args.model == 'dnn':
        result = run_dnn(X_train, X_test, y_train, y_test, input_dim=X_train.shape[1], return_probs=True, epochs=args.epochs)
        results.append(result['metrics'])
        all_probs['DNN'] = result['probs']
        torch.save(result['model'].state_dict(), "artifacts/dnn_model.pt")

    elif args.model == 'all':
        result_lr = run_lr(X_train, X_test, y_train, y_test, return_probs=True)
        results.append(result_lr['metrics'])
        all_probs['Logistic Regression'] = result_lr['probs']
        joblib.dump(result_lr['model'], "artifacts/lr_model.pkl")

        result_xgb = run_xgb(X_train, X_test, y_train, y_test, return_probs=True)
        results.append(result_xgb['metrics'])
        all_probs['XGBoost'] = result_xgb['probs']
        joblib.dump(result_xgb['model'], "artifacts/xgb_model.pkl")

        result_dnn = run_dnn(X_train, X_test, y_train, y_test, input_dim=X_train.shape[1], return_probs=True, epochs=args.epochs)
        results.append(result_dnn['metrics'])
        all_probs['DNN'] = result_dnn['probs']
        torch.save(result_dnn['model'].state_dict(), "artifacts/dnn_model.pt")

    # Print and save results
    df_results = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(df_results)
    save_metrics_to_csv(results)
    plot_model_metrics_histogram()

    # Plot AUC curve
    if all_probs:
        plot_auc_curve(all_probs, y_test)
