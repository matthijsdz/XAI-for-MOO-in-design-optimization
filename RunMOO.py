import numpy as np
import pandas as pd
from MOOCourse.modules.utils import load

from WBProblem import WBProblem
from TBTProblem import TBTProblem
from Helper import ScatterPlot

import matplotlib.pyplot as plt

def create_random_data(problem, n_variables):
    nr_points = 10000
    bounds = problem.get_variable_bounds()
    var = []
    for i in range(nr_points):
        point = np.zeros(n_variables)
        for nr  in range(n_variables):
            point[nr] = np.random.uniform(bounds[nr][0], bounds[nr][1])
        var.append(point)
    obj = problem.evaluate_objectives(var)[0]
    data = np.concatenate((obj,var), axis=1)
    return data

def main():
    ProblemType = "TBT"

    # Find pareto front

    MOOproblem = TBTProblem()

    print("optimize")

    ParetoFront = MOOproblem.optimize()

    if ProblemType == "TBT":
        ScatterPlot(ParetoFront[:,0],ParetoFront[:,1],ParetoFront[:,2])
    else:
        ScatterPlot(ParetoFront[:,0],ParetoFront[:,1])

    # create random datapoints
    print("create random points")

    variable_names = MOOproblem.problem.get_variable_names()
    objective_names = MOOproblem.problem.get_objective_names()
    RandomPoints = create_random_data(MOOproblem.problem, len(variable_names))
    CompleteDataset = np.concatenate((ParetoFront,RandomPoints), axis=0)

    if ProblemType == "TBT":
        ScatterPlot(CompleteDataset[:,0],CompleteDataset[:,1],CompleteDataset[:,2])
    else:
        ScatterPlot(CompleteDataset[:,0],CompleteDataset[:,1])
    columns = objective_names + variable_names
    CompleteDataset = pd.DataFrame(CompleteDataset,columns=columns)
    CompleteDataset.to_csv("datasets/{}.csv".format(ProblemType), index=False)
    # labelling the dataset
    print("labelling dataset")

    columns = objective_names + variable_names + ['label', 'label_names']
    LabelledDataset = MOOproblem.add_labels(CompleteDataset, columns)
    LabelledDataset.to_csv("datasets/labelled_{}.csv".format(ProblemType), index=False)





if __name__ == "__main__":
    main()
