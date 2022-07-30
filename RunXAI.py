import pandas as pd
import numpy as np
import scipy.stats

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import statistics
from XAIMethods import LIME, SHAP, PDP, PFI

import warnings

warnings.filterwarnings("ignore")

def load_data(filename, ProblemType):
    df = pd.read_csv('Datasets\{}.csv'.format(filename))
    if ProblemType == "WB":
        X = df.iloc[:,2:6]
        class_names = ["KP", "F1", "F2", "BD"]
    if ProblemType == "TBT":
        X = df.iloc[:,3:9]
        class_names = ["KP", "F1", "F2", "F3", "F12", "F23", "F13", "BD"]
    y = df['label']
    return X,y, class_names

def RunRandomForest(X,y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    classifier = RandomForestClassifier(n_estimators = 100, criterion='entropy', max_depth=1000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("accuracy = {}".format(metrics.accuracy_score(y_test, y_pred)))
    print(pd.crosstab(y_test, y_pred, rownames=['Actual Classes'], colnames=['Predicted Classes']))
    return classifier

def explain(classifier, X, y, class_names,XAIMethod,ProblemType):

    if XAIMethod == "PFI":
        explainer = PFI(classifier, X, y, class_names, ProblemType)
        explainer.plot(filename="{}_PFI_importances".format(ProblemType))

    if XAIMethod == "PDP":
        explainer = PDP(classifier, X, y, class_names, ProblemType)
        explainer.plot(filename="{}_PDP_importances_mean".format(ProblemType))

    if XAIMethod == "SHAP":
        explainer = SHAP(classifier, X, y, class_names, ProblemType)
        explainer.plot_total_summary(filename="{}_SHAP_total_summaryplot".format(ProblemType))
        explainer.plot_summary_per_class(filename="{}_SHAP_summaryplot_per_class".format(ProblemType))
        explainer.plot_force_plot(i=5,filename="{}_SHAP_forceplot".format(ProblemType))

    if XAIMethod == "LIME":
        explainer = LIME(classifier, X, y, class_names, ProblemType)
        explainer.plot(i=5, filename="{}_LIME_test".format(ProblemType))

def experiment(ProblemType):
    X,y,class_names = load_data("labelled_{}".format(ProblemType), ProblemType)
    classifier = RunRandomForest(X,y)
    methods = ["PFI", "PDP", "SHAP", "LIME"]
    for method in methods:
        explain(classifier, X, y, class_names,  method, ProblemType)

if __name__ == "__main__":
#    ProblemTypes = ["TBT",  "WB"]
    ProblemTypes = ["TBT"]

    for ProblemType in ProblemTypes:
        experiment(ProblemType)
