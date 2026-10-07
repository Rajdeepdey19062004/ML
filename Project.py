import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


FEATURES = ["radius_mean", "texture_mean"]
TARGET = "diagnosis"


def evaluate_models(dataset_path: Path, output_dir: Path, show_plots: bool) -> None:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))

    data = data[FEATURES + [TARGET]].copy()
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna(subset=FEATURES + [TARGET])
    if not data[TARGET].isin(["B", "M"]).all():
        raise ValueError("Diagnosis must contain only 'B' and 'M' values.")

    class_counts = data[TARGET].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("At least two samples of each diagnosis class are required.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="linear", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    labels = ["B", "M"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))

    for axis, (name, classifier) in zip(axes.flat, models.items(), strict=True):
        model = make_pipeline(StandardScaler(), classifier)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f"\n{name} accuracy: {accuracy * 100:.2f}%")
        print(
            classification_report(
                y_test,
                predictions,
                labels=labels,
                target_names=["Benign", "Malignant"],
                zero_division=0,
            )
        )

        matrix = confusion_matrix(y_test, predictions, labels=labels)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=axis,
        )
        axis.set_title(f"{name}\nAccuracy: {accuracy:.1%}")
        axis.set_xlabel("Predicted diagnosis")
        axis.set_ylabel("Actual diagnosis")

    figure.suptitle("Breast Cancer Classifier Comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "model_confusion_matrices.png", dpi=200)
    if show_plots:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare classifiers on the breast cancer dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().with_name("cancer_data.csv"),
        help="CSV dataset path (default: cancer_data.csv beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for generated charts (default: current directory).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save charts without opening a plot window.",
    )
    args = parser.parse_args()
    evaluate_models(args.dataset.expanduser().resolve(), args.output_dir, not args.no_show)


if __name__ == "__main__":
    main()
