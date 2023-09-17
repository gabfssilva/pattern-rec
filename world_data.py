import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import seaborn as sns


def load_data(filepath):
    return pd.read_csv(filepath)


def exploratory_analysis(df):
    return # just to avoid plotting stuff all the time

    print(df.describe())
    print(df.info())

    sns.pairplot(df)
    plt.show()


def preprocess_data(df):
    df['GDP'] = df['GDP'].str.replace('$', '').str.replace(',', '').astype(float)
    df.fillna(method='ffill', inplace=True)
    # for col in df.select_dtypes(include=['float64', 'int64']).columns:
    #     df[col].fillna(df[col].mean(), inplace=True)
    bins = [0, 1e10, 1e11, float('inf')]
    labels = ['baixo', 'médio', 'alto']
    df['GDP_category'] = pd.cut(df['GDP'], bins=bins, labels=labels)
    category_copy = df['GDP_category'].copy()
    hot_encoded = pd.get_dummies(df, drop_first=False)
    hot_encoded['GDP_category'] = category_copy

    return hot_encoded


def dimensionality_reduction(X_train, X_test):
    pca = PCA()
    pca.fit(X_train)

    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_cumulative = np.cumsum(explained_variance_ratio)

    ideal_pca_components = np.where(explained_variance_cumulative > 0.9)[0][0] + 1

    pca = PCA(n_components=ideal_pca_components)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def choose_models():
    return [DecisionTreeClassifier(), GaussianNB(), SVC(), KNeighborsClassifier(n_neighbors=10)]


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  # Usando média ponderada para multiclasse
    return accuracy, precision


def main():
    df = load_data('world-data-2023.csv')
    exploratory_analysis(df)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['GDP', 'GDP_category', 'GDP_category_baixo', 'GDP_category_médio', 'GDP_category_alto'], axis=1),
        df['GDP_category'].copy(), test_size=0.1, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    models = choose_models()
    model_names = ["Decision Tree", "Naive Bayes", "SVM", "KNN"]

    for model, name in zip(models, model_names):
        trained_model = train_model(model, X_train, y_train)
        accuracy, precision = evaluate_model(trained_model, X_test, y_test)
        print(f"Accuracy & precision {name}: {accuracy} -- {precision}")

    X_train, X_test = dimensionality_reduction(X_train, X_test)

    for model, name in zip(models, model_names):
        trained_model = train_model(model, X_train, y_train)
        accuracy, precision = evaluate_model(trained_model, X_test, y_test)
        print(f"[PCA] Accuracy & precision {name}: {accuracy} -- {precision}")

if __name__ == "__main__":
    main()
