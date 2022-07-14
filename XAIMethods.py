import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error

import lime.lime_tabular
import shap

from IPython.display import HTML

class PFI:
    def __init__(self, classifier, X, y, class_names, ProblemType, n_repeats = 100):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.feature_names = list(X.columns)
        self.class_names = class_names
        self.n_repeats = n_repeats
        self.ProblemType = ProblemType

    def plot(self, filename="PFI_importances"):
        FI = []
        scaler = StandardScaler()
        predictions = self.classifier.predict(scaler.fit_transform(self.X))
        e_orig = mean_squared_error(predictions, self.y)
        for j in self.feature_names:
            importances = []
            for k in range(self.n_repeats):
                X_perm = self.X.copy()
                X_perm[j] = np.random.permutation(X_perm[j].values)
                predictions = self.classifier.predict(scaler.fit_transform(X_perm))
                e_perm = mean_squared_error(self.y,predictions)
                importances.append((e_perm)/e_orig)
            FI.append(importances)
        plt.boxplot(FI, vert=False, labels=self.feature_names)
        plt.title("Feature Premutation Importance per feature")
        plt.xlabel("Importance")
        plt.ylabel("features")
        plt.savefig("Plots/{0}".format(filename),dpi=300)
        plt.show()
        pass

class LIME():
    def __init__(self, classifier, X, y, class_names, ProblemType):
        self.feature_names = list(X.columns)
        self.class_names = class_names
        self.X = X.to_numpy()
        self.y = y.tolist()
        self.classifier = classifier
        self.ProblemType = ProblemType
        self.explainer = lime.lime_tabular.LimeTabularExplainer(X.to_numpy(),feature_names=self.feature_names,class_names=self.class_names, mode='classification')
        self.explanation = None

    def plot(self,i=5,filename="LIME_test"):
        self.explanation = self.explainer.explain_instance(self.X[i],self.classifier.predict_proba,top_labels=8,num_features=8)
        self.explanation.save_to_file("plots\{0}".format(filename))
        print("LIME plot saved")

class SHAP():
    def __init__(self, classifier, X, y, class_names, ProblemType):
        self.explainer = shap.TreeExplainer(classifier, X)
        shap.initjs()
        self.shap_values = self.explainer.shap_values(X)
        self.feature_names = list(X.columns)
        self.class_names = class_names
        self.X = X
        self.classifier = classifier
        self.plot = None
        self.ProblemType = ProblemType

    def plot_total_summary(self, filename="SHAP_total_summaryplot"):
        fig = shap.summary_plot(self.shap_values, self.X, feature_names=self.feature_names,class_names=self.class_names, show=False, plot_size = (12,6))
        plt.savefig("Plots/{0}".format(filename))
        print("SHAP_total_summary_plot saved")
        plt.show()

    def plot_summary_per_class(self, filename="SHAP_total_summaryplot_per_class"):
        for j in range(len(self.class_names)):
            fig = shap.summary_plot(self.shap_values[j], self.X, feature_names=self.feature_names,class_names=self.class_names, show=False,plot_size = (12,5))
            plt.title("{}".format(self.class_names[j]))
            plt.savefig("Plots/{0}_{1}".format(filename,self.class_names[j]), dpi=300)
            print("SHAP_summary_plot_{} saved".format(self.class_names[j]))
            plt.show()

    def plot_force_plot(self,i=5, filename="SHAP_forceplot"):
        X = self.X.to_numpy()
        max_exp_value = np.argmax(self.explainer.expected_value)
        shap.force_plot(self.explainer.expected_value[max_exp_value], self.shap_values[max_exp_value][i],
        X[i].round(3), feature_names=self.feature_names,matplotlib=True, show=False).savefig("Plots/{0}_{1}".format(filename,i),dpi=300)
        print("SHAP_forceplot plot saved")
        plt.show()

class PDP():
    def __init__(self, classifier, X, y, class_names, ProblemType):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.feature_names = list(X.columns)
        self.class_names = class_names
        self.ProblemType = ProblemType
        self.fig,self.ax = plt.subplots(1, int(len(self.feature_names)))

    def plot(self, filename="PDP_test"):
        features = np.arange(len(self.feature_names))
        pdp1 = PartialDependenceDisplay.from_estimator(self.classifier, self.X, features = features,
                feature_names=self.feature_names,target=0, ax=self.ax, line_kw={"label": self.class_names[0],"color": "red"})
        pdp2 = PartialDependenceDisplay.from_estimator(self.classifier, self.X, features = features,
                feature_names=self.feature_names,target=1, ax=pdp1.axes_, line_kw={"label": self.class_names[1], "color": "blue"})
        pdp3 = PartialDependenceDisplay.from_estimator(self.classifier, self.X, features = features,
                feature_names=self.feature_names,target=2, ax=pdp1.axes_, line_kw={"label": self.class_names[2], "color": "green"})
        if self.ProblemType == "TBT":
            pdp4 = PartialDependenceDisplay.from_estimator(self.classifier, self.X, features = features,
                feature_names=self.feature_names,target=3, ax=pdp1.axes_, line_kw={"label": self.class_names[3], "color": "orange"})
        #self.ax.legend()
        #self.fig.savefig("Plots/{0}".format(filename),dpi=300)
        plt.show()
        print("PDP plot saved")
