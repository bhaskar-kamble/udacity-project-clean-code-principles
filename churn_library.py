"""
This file is part of the project submission for Udacity's course on Clean Coding Principles.

It contains refactored code of churn data analysis adhering to clean coding principles.

It carries out exploratory data analysis and fits machine learning models to predict 
when a customer is likely to churn.

Files and models are saved in 'images' and 'models' folders respectively.

This can be tested with churn_script_logging_and_tests.py.

Author: Bhaskar Kamble
Date: April 5, 2025
"""
import os

import joblib
from sklearn.metrics import classification_report, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import keep_cols, category_lst, input_file_path, response, \
    lr_model_path, rfc_model_path, use_saved_models


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set_theme()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_from_csv = pd.read_csv(pth)
    return df_from_csv


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    filename = "./images/eda/churn_distribution.png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    filename = "./images/eda/customer_age_distribution.png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 10))
    filename = "./images/eda/marital_status_distribution.png"
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    filename = "./images/eda/total_transaction_distribution.png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 10))
    use_original = False
    if not use_original:
        sns.heatmap(
            df.drop(
                category_lst +
                ["Attrition_Flag"],
                axis=1).corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
    else:
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    filename = "./images/eda/heatmap.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def encoder_helper(df, category_list, rspnse):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_list: list of columns that contain categorical features
            rspnse: string of response name [optional argument that could
                      be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df[rspnse] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    for cat in category_list:
        cat_groups = df.groupby(cat)[rspnse].mean()
        df[cat + "_" + rspnse] = df[cat].apply(lambda x: cat_groups.loc[x])

    return df


def perform_feature_engineering(df, rspnse):
    '''
    input:
              df: pandas dataframe
              rspnse: string of response name
              [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    df = encoder_helper(df, category_lst, rspnse)
    X = pd.DataFrame()
    y = df[rspnse]
    X[keep_cols] = df[keep_cols]
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return train_x, test_x, train_y, test_y


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.95, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.ylim(0.3, 1.3)
    plt.axis('off')
    plt.savefig("./images/results/rf_results.png")
    plt.close()

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.95, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.ylim(0.3, 1.3)
    plt.axis('off')
    plt.savefig("./images/results/logistic_results.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches="tight")
    plt.close()


def plot_both_roc_curves(
        clf1,
        clf1_name,
        clf2,
        clf2_name,
        features,
        target,
        filepath):
    """
    PUT DOCUMENTATION HERE
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    predictions = clf1.predict_proba(features)
    predictions = predictions[:, 1]
    roc_auc_1 = roc_auc_score(target, predictions)
    RocCurveDisplay.from_predictions(
        target, predictions, color="blue", marker="o", ax=ax)

    predictions = clf2.predict_proba(features)
    predictions = predictions[:, 1]
    roc_auc_2 = roc_auc_score(target, predictions)
    RocCurveDisplay.from_predictions(
        target, predictions, color="red", marker="o", ax=ax)

    ax.legend([f'{clf1_name} (AUC = {roc_auc_1:.2f})',
               f'{clf2_name} (AUC = {roc_auc_2:.2f})'], loc="lower right")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    plt.savefig(filepath)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # logistic regression
    lrc = LogisticRegression(solver='newton-cholesky', max_iter=3000)  # lbfgs
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # grid search for random forest
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    plot_both_roc_curves(
        lrc,
        "LogisticRegression",
        cv_rfc.best_estimator_,
        "RandomForest",
        X_test,
        y_test,
        "./images/results/roc_curve_result.png")

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc.best_estimator_, X_test,
                            "./images/results/feature_importances.png")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, rfc_model_path)
    joblib.dump(lrc, lr_model_path)


if __name__ == "__main__":
    df_data = import_data(input_file_path)
    perform_eda(df_data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_data, response)

    # if you have to train models from scratch:
    # use_saved_models = True

    if use_saved_models:
        lr_model = joblib.load(lr_model_path)
        preds_train_lr = lr_model.predict(X_train)
        preds_test_lr = lr_model.predict(X_test)

        rfc_model = joblib.load(rfc_model_path)
        preds_train_rf = rfc_model.predict(X_train)
        preds_test_rf = rfc_model.predict(X_test)

        classification_report_image(y_train,
                                    y_test,
                                    preds_train_lr,
                                    preds_train_rf,
                                    preds_test_lr,
                                    preds_test_rf)

        plot_both_roc_curves(lr_model, "LogisticRegression",
                             rfc_model, "RandomForest", X_test, y_test,
                             "./images/results/logreg_rf_roc.png")

        feature_importance_plot(rfc_model,
                                X_test,
                                "./images/results/feature_importances.png")
    else:
        train_models(X_train, X_test, y_train, y_test)
