import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_dataset(choice):
    if choice == "iris":
        data = load_iris()
    elif choice == "wine":
        data = load_wine()
    elif choice == "cancer":
        data = load_breast_cancer()
    else:
        print("Invalid choice. Defaulting to Iris.")
        data = load_iris()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def visualize_data(df):
    print("\nShowing pairplot visualization...")
    sns.pairplot(df, hue="target")
    plt.show()


def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)


if __name__ == "__main__":
    print("Choose dataset: iris / wine / cancer")
    choice = input("Enter dataset name: ").lower()

    df = load_dataset(choice)
    print("\nFirst 5 rows:")
    print(df.head())

    visualize_data(df)
    train_model(df)
