#  Phishing Detection with Machine Learning & Deep Learning

This project builds and compares multiple models (Logistic Regression, XGBoost, and a PyTorch DNN) for detecting phishing emails using TF-IDF and a RESTful FastAPI interface for real-time inference.

---

##  Features

- Text preprocessing (cleaning + TF-IDF vectorization)
- Model training & evaluation (accuracy, precision, recall, F1, AUC)
- Confusion matrix and metric visualizations
- Trained model + vectorizer saving (artifacts/)
- REST API with FastAPI + HTML UI for inference

---

##  Project Structure

```
.
├── main.py                  # Training entrypoint
├── inference_api.py         # FastAPI inference server
├── templates/
│   └── form.html            # HTML UI for prediction
├── utils/
│   └── processing.py        # Cleaning, preprocessing, metrics, plots
├── Models/
│   └── trainers.py          # Training code for LR, XGB, DNN
├── artifacts/               # Saved models and vectorizer
└── requirements.txt         # Python dependencies
```

---

##  Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Training Models

Run training using:

```bash
python main.py --model all --data data/phishing_dataset.csv --epochs 10
```

Supported values for `--model`:
- `lr` : Logistic Regression
- `xgb` : XGBoost
- `dnn` : Deep Neural Network (PyTorch)
- `all` : Train and evaluate all

---

##  Outputs

After training, check:
- ROC curve: `outputs/auc_curves.png`
- Confusion matrix: `outputs/confusion_matrix_*.png`
- Model metrics: `outputs/model_metrics.csv`
- Saved models: `artifacts/`

---

## Running the API

```bash
python -m uvicorn inference_api:app --reload
```

Then visit:
- HTML Form: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

Phishing Email Dataset (Kaggle):  
[https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)


---

## Author

Built with by **Tanayendu Bari**  
