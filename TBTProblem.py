"""
Contains the implementatiom of the Two Bar Truss problem using functions from
MOOCourse.
"""

from MOOCourse.modules.TwoBarTruss.problem import create_problem
from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class TBTProblem():

    def __init__(self):
        self.load = 65
        # weight, stress, buckling stress and deflection
        self.obj = np.array([True, True, True, False,])

        self.constraints = np.array([
            [10, 100], #  10 < weight < 100
            [15, None], # stress > 15
            [None, 100], # buckling < 100
            [None, None], # deflection no constraint
        ])

        self.problem,self.method =  create_problem(self.load, self.obj, self.constraints)

    def optimize(self):
        step_sizes = np.array([10, 77, 10, 4])[self.obj]

        var, obj = solve_pareto_front_representation(self.problem, step_sizes)

        data = np.concatenate((obj,var), axis=1)

        return data

    def add_labels(self, data, columns, filename='labelled_TBT.csv'):
        new_data = []
        for row in data:
            row = row.tolist()
            if row[0] <= 10 and row[1] <= 100 and row[2] <= 10:        #KP
                row.extend([0, "KP"])
                new_data.append(row)
            if row[0] <= 10 and row[1] > 120 and row[1] <= 220  and row[2] > 10 and row[2] < 30:     #F1
                row.extend([1, "F1"])
                new_data.append(row)
            if row[0] > 10 and row[0] > 30 and row[1] <= 100 and row[2] > 10 and row[2] <= 30:      #F2
                row.extend([2, "F2"])
                new_data.append(row)
            if row[0] > 10 and row[0] < 30 and row[1] > 120 and row[1] < 180 and row[2] <= 10:     #F3
                row.extend([3, "F3"])
                new_data.append(row)
            if row[0] <= 10 and row[1] <= 100 and row[2] > 10 and row[2] < 30:       #F12
                row.extend([4, "F12"])
                new_data.append(row)
            if row[0] > 10 and row[1] <= 100 and row[2] <= 10:       #F23
                row.extend([5, "F23"])
                new_data.append(row)
            if row[0] <= 10 and row[1] > 120 and row[2] <= 10:       #F13
                row.extend([6, "F13"])
                new_data.append(row)
            if row[0] > 10 and row[1] > 120 and row[2] > 10:     #BD
                row.extend([7, "BD"])
                new_data.append(row)
        df = pd.DataFrame(new_data,columns=columns)
        return df
