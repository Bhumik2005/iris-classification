import argparse
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def visualize_data(df):
    print("Showing dataset visualization...")
    sns.pairplot(df, hue="target")
    plt.show()


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{name} Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{name} Confusion Matrix")
        plt.show()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model


def save_model(model):
    joblib.dump(model, "model/best_model.pkl")
    print("\nBest model saved to model/best_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    df = load_data()

    if args.visualize:
        visualize_data(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    best_model = evaluate_models(models, X_test, y_test)
    save_model(best_model)
