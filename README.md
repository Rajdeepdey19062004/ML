# Breast Cancer Classifier Comparison

This project compares six classifiers on the Wisconsin Diagnostic Breast
Cancer dataset using `radius_mean` and `texture_mean` to predict the `diagnosis`
label (`B` = benign, `M` = malignant). It reports accuracy and classification
metrics and saves a separate confusion-matrix plot for each model.

## Requirements

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run

From this directory:

```bash
python Project.py --no-show
```

The script reads `cancer_data.csv` beside itself by default and writes
`model_confusion_matrices.png` to the current directory. You can specify
different paths with `--dataset` and `--output-dir`; use `--help` for details.

The data is split with stratification and a fixed random seed for reproducible
evaluation. Standardization is fitted only on the training split through a
scikit-learn pipeline.

## Project files

```text
Project.py
cancer_data.csv
requirements.txt
README.md
```

## Educational use

This is a learning and model-comparison exercise, not a medical diagnostic
tool. Classification performance on this dataset does not establish clinical
validity.

The repository's earlier README attributed an associated web project to
Sayan Mahalanabish; that attribution is retained here as historical credit.
