import math

import pandas
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    explained_variance_score, r2_score, mean_squared_log_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def evaluate_preds_classification(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1} \n")
    return metric_dict


def evaluate_preds_regression(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    """
    r2 = r2_score(y_true, y_preds)
    mse = mean_squared_error(y_true, y_preds)
    mae = mean_absolute_error(y_true, y_preds)
    evs = explained_variance_score(y_true, y_preds)
    msle = mean_squared_log_error(y_true, y_preds)
    rmsle = math.sqrt(msle)
    metric_dict = {"Coefficient of determination": r2,
                   "MSE": mse,
                   "MAE": mae,
                   "MSLE": msle,
                   "RMSLE": rmsle,
                   "Explained Variance score": evs
                   }

    print(f"Coefficient of determination: {r2}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MSLE: {msle}")
    print(f"RMSLE: {rmsle}")
    print(f"Explained Variance score: {evs}")
    return metric_dict


def fit_and_score(models: dict, x_train, x_test, y_train, y_test, type_of_problem: str, plot: bool = False) -> dict:
    """
    Fits and evaluates given machine learning models.

    :param type_of_problem: 'regression' or 'classification'
    :param models: a dict of different Scikit-Learn machine learning models
    :param x_train: training data (no target values)
    :param x_test: testing data (no target values)
    :param y_train: training target values
    :param y_test: test target values
    :param plot : boolean - If true, also plot will be displayed. False on default.
    :return: dictionary containing scores of the models.
    """

    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    scores = {}
    # Loop through models
    for name, model in models.items():
        model.fit(x_train, y_train)
        # Evaluate the model and append it's score to the scores dict
        y_preds = model.predict(x_test)
        print(f"========== {name} ==========")
        if type_of_problem == "classification":
            scores[name] = evaluate_preds_classification(y_test, y_preds)
        elif type_of_problem == "regression":
            scores[name] = evaluate_preds_regression(y_test, y_preds)
        else:
            raise ValueError("type_of_problem must be either 'regression' or 'classification'")
    if plot:
        model_compared = pd.DataFrame(scores.values(), index=models.keys())
        model_compared.plot.bar(figsize=(10, 8))
    return scores


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().

    :param y_test: True target values
    :param y_preds: Predicted target values
    :return: Matmplotlib's figure and axis
    """
    sns.set(font_scale=1.5)  # Increase font size
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,  # Annotate the boxes
                     cbar=False)
    plt.xlabel("Predicted label")  # predictions go on the x-axis
    plt.ylabel("True label")  # true labels go on the y-axis
    return fig, ax


def plot_correlation_mat(corr_matrix):
    """
    Plots Correlation matrix using Seaborn's heatmap()
    :param corr_matrix: Pandas Correlation matrix
    :return: Figure and axis of the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    return fig, ax


def cross_validated_report_classification(clf, x, y, plot=False):
    """

    :param clf: Scikit-Learn's classifier
    :param x:   Features
    :param y:   Targets
    :param plot: boolean - If true plot is also being draw. False by default.
    :return:    Pandas DataFrame with containing Cross-validated Accuracy, Precision, Recall, F1 in this order.
    """
    # Cross-validated accuracy
    acc = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="accuracy"))
    print(f"Cross-validated accuracy: {acc}")

    # Cross-validated precision
    precision = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="precision"))
    print(f"Cross-validated precision: {precision}")

    # Cross-validated recall
    rec = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="recall"))
    print(f"Cross-validated recall: {rec:.2f}")

    # Cross-validated F1 score
    f1 = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="f1"))
    print(f"Cross-validated F1 Score: {f1}")

    metrics = pd.DataFrame({"Accuracy": acc,
                            "Precision": precision,
                            "Recall": rec,
                            "F1 Score": f1}, index=[0])

    if plot:
        metrics.T.plot.bar(title="Cross-Validated model metrics",
                           legend=False, yticks=np.arange(0, 1, 0.1))

    return metrics


def cross_validated_report_regression(clf, x, y, plot=False):
    """

    :param clf: Scikit-Learn's classifier
    :param x:   Features
    :param y:   Targets
    :param plot: boolean - If true plot is also being draw. False by default.
    :return:    Pandas DataFrame with containing Cross-validated metrics.
    """
    r2 = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="r2"))
    print(f"Cross-validated Coefficient of Determination: {r2}")

    mse = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="neg_mean_squared_error"))
    print(f"Cross-validated MSE {mse}")

    mae = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="neg_mean_absolute_error"))
    print(f"Cross-validated MAE: {mae}")

    evs = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="explained_variance"))
    print(f"Cross-validated F1 Score: {evs}")

    msle = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="neg_mean_squared_log_error"))
    print(f"Cross-validated F1 Score: {evs}")

    rmsle = mean_and_round(math.sqrt(msle))
    print(f"Cross-validated F1 Score: {evs}")

    metrics = pd.DataFrame({"Coefficient of determination": r2,
                            "MSE": mse,
                            "MAE": mae,
                            "MSLE": msle,
                            "RMSLE": rmsle,
                            "Explained Variance score": evs
                            }, index=[0])

    if plot:
        metrics.T.plot.bar(title="Cross-Validated model metrics",
                           legend=False, yticks=np.arange(0, 1, 0.1))

    return metrics


def mean_and_round(np_arr):
    """
    Takes an NumPy objects and calculate mean rounded up to 2 decimal places. Uses NumPy.
    :param np_arr: iterable sequence of numbers
    :return: rounded mean of values in iterable
    """

    return np.around(np.mean(np_arr), 2)


def add_datetime_parameters(df: pandas.DataFrame, name_of_ts_column: str, drop_column: bool = False):
    """
    Adds datetime parameters to provided Pandas DataFrame in place
    :param drop_column: If true column Timestamp object will be dropped, False by default.
    :param df: Pandas DataFrame to enrich
    :param name_of_ts_column: name of the column containing Pandas Timestamp object
    :return: None
    """
    df["saleMonth"] = df[name_of_ts_column].dt.month
    df["saleYear"] = df[name_of_ts_column].dt.year
    df["saleDay"] = df[name_of_ts_column].dt.day
    df["saleDayOfWeek"] = df[name_of_ts_column].dt.day_of_week
    df["saleDayOfYear"] = df[name_of_ts_column].dt.day_of_year

    if drop_column:
        df.drop(name_of_ts_column, axis=1, inplace=True)


def find_string_columns(df: pandas.DataFrame, verbose: bool = False):
    """
    Utility function that finds columns containing string.
    :param verbose: If true, found labels will be printed, false by default
    :param df: Pandas dataframe
    :return: list of column's labels containing string data
    """
    labels = []
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            labels.append(label)
            if verbose:
                print(label)

    return labels


def find_numeric_columns(df: pandas.DataFrame, verbose: bool = False):
    """
    Utility function that finds columns containing string.
    :param verbose: If true, found labels will be printed, false by default
    :param df: Pandas dataframe
    :return: list of column's labels containing string data
    """
    labels = []
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            labels.append(label)
            if verbose:
                print(label)

    return labels


def find_categorical_columns(df: pandas.DataFrame, verbose: bool = False):
    """
    Utility function that finds columns containing string.
    :param verbose: If true, found labels will be printed, false by default
    :param df: Pandas dataframe
    :return: list of column's labels containing string data
    """
    labels = []
    for label, content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            labels.append(label)
            if verbose:
                print(label)

    return labels


def convert_strings_into_categories(df: pandas.DataFrame):
    """
    Utility function that finds columns containing string and convert them into categories.
    :param df: Pandas dataframe
    :return: None
    """
    for label in find_string_columns(df):
        df[label] = df[label].astype("category").cat.as_ordered()


def plot_feature_importance(columns, importances, n=20):
    """

    :param columns: sequence of column names
    :param importances: sequence of features importances
    :param n: number of features to display
    :return: tuple with fig, ax
    """
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    # make plot
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    return fig, ax
