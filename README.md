# XAI-for-MOO-in-building-design
Contains code used for the thesis "Explainable AI in Multi-Objective Design Optimization"
***
## Files
* Helpler.py: contains helper function for vizualising optimized pareto front
* TBTProblem.py: Implementation of the two-bar truss problem and function for transforming the optimization dataset into a classification data set
* WBProblem.py: Implementation of the welded beam problem and function for transforming the optimization dataset into a classification data set
* RunMOO.py: code for running multi-objective optimization algorithm on welded beam or two-bar truss problem.
* RunXAI.py: runs Permutation Feature Importance (PFI), Partial dependence Plots(PDPs), SHAP and LIME methods on the optimization data
* XAIMethods.py contains Permutation Feature Importance (PFI), Partial dependence Plots(PDPs), SHAP and LIME methods
