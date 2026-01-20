
# Automated Machine Learning Pipeline using DVC

An end-to-end automated Machine Learning pipeline built using **Python**, **Scikit-learn**, and **DVC**, ensuring full reproducibility and version control of data, models, and experiments.

---

## Overview

This project automates the complete Machine Learning workflow:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Metrics and artifact tracking using DVC

The pipeline is modular, reproducible, and easy to maintain.

---

## Problem Statement

Predict medical insurance charges based on customer attributes using supervised machine learning (Linear Regression).

---

## Dataset Description

Features included:
- age
- sex
- bmi
- children
- smoker
- region
- charges (target)

---

## Project Structure

```

.
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── model.pkl
├── reports/
│   └── test_metrics.json
├── dvc.yaml
├── dvc.lock
├── config.yaml
├── params.yaml
├── .gitignore
├── .dvcignore
├── requirements.txt
└── README.md

````

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- DVC
- Git

---

## How to Run the Pipeline

Install dependencies:
```bash
pip install -r requirements.txt
````

Run the pipeline:

```bash
dvc repro
```

View metrics:

```bash
cat reports/test_metrics.json
```

---

## Version Control Strategy

* Git tracks source code and configuration files
* DVC tracks datasets, models, and metrics
* Large files are not committed to Git directly

---

## Future Improvements

* Hyperparameter tuning
* Advanced regression models
* Cross-validation
* MLflow integration
* API deployment

---