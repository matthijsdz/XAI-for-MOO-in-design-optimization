"""
Contains the implementatiom of the Two Bar Truss problem using functions from
MOOCourse.
"""

import numpy as np
import pandas as pd
import warnings

from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem.problem.Objective import _ScalarObjective
from desdeo_problem.problem.Variable import variable_builder
from desdeo_mcdm.utilities.solvers import payoff_table_method
from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from scipy.optimize import  minimize
from desdeo_tools.solver import ScalarMethod
from desdeo_problem import (
    VectorObjective,
    Variable,
    ScalarConstraint,
    MOProblem
)

warnings.filterwarnings("ignore")


class TBTProblem():

    def __init__(self):
        self.load = 65
        # weight, stress, buckling stress and deflection
        self.obj = np.array([True, True, True, False,])
        self.initial_values = None
        self.objectives, self.obj_f = self.create_objectives()
        self.constraints = self.create_constraints()
        self.variables = self.create_variables()
        self.problem,self.method = self.create_problem()

    #***************************************************************************
    #desdeo: JSS 2021 Summer School, Jyvaskyla, Finland
    #(Michael Emmerich, Bhupinder Saini)

    def constraint_builder(self, f, n_obj, n_var, bound, is_lower_bound = True, name= "c1"):
        c = lambda xs, _ys: f(xs) - bound if is_lower_bound else bound - f(xs)
        return ScalarConstraint(name, n_var, n_obj, c)

    def create_objectives(self, obj_mask=[True, True, True, False]):
        if type(obj_mask) is not np.ndarray:
            obj_mask = np.array(obj_mask)

        def weight(xs: np.ndarray) -> np.ndarray:
            xs = np.atleast_2d(xs)
            H, d, t, B, E, p = xs.T  # Assign the values to named variables for clarity
            return p * 2 * np.pi * d * t * np.sqrt(np.square(B / 2) + np.square(H))

        def stress(xs: np.ndarray) -> np.ndarray:
            xs = np.atleast_2d(xs)
            H, d, t, B, E, p = xs.T
            numerator = self.load * np.sqrt(np.square(B / 2) + np.square(H))
            denominator = 2 * t * np.pi * d * H
            return numerator / denominator

        def buckling_stress(xs: np.ndarray) -> np.ndarray:
            xs = np.atleast_2d(xs)
            H, d, t, B, E, p = xs.T
            numerator = np.square(np.pi) * E * (np.square(d) + np.square(t))
            denominator = 8 * (np.square(B / 2) + np.square(H))
            return numerator / denominator

        def deflection(xs: np.ndarray) -> np.ndarray:
            xs = np.atleast_2d(xs)
            H, d, t, B, E, p = xs.T
            numerator = self.load * np.power(np.square(B / 2) + np.square(H), (3/2))
            denominator = 2 * t * np.pi * d * np.square(H) * E
            return numerator / denominator

        # Define objectives
        obj1 = _ScalarObjective("Weight", weight)
        obj2 = _ScalarObjective("Stress", stress)
        obj3 = _ScalarObjective("Buckling stress", buckling_stress)
        obj4 = _ScalarObjective("Deflection", deflection)
        objectives = np.array([obj1, obj2, obj3, obj4])[obj_mask]

        obj_f = [weight, stress, buckling_stress, deflection]

        return objectives, obj_f

    def create_constraints(self):
        constraints = np.array([
                    [10, 100], #  10 < weight < 100
                    [15, None], # stress > 15
                    [None, 100], # buckling < 100
                    [None, None], # deflection no constraint
        ])
        n_objectives = 3
        n_variables = 6
        cons = []
        for i in range(4):
            lower, upper = constraints[i]
            if lower is not None:
                con = self.constraint_builder(self.obj_f[i], n_objectives, n_variables, lower, True, f"c{i}l")
                cons.append(con)
            if upper is not None:
                con = self.constraint_builder(self.obj_f[i], n_objectives, n_variables, upper, False, f"c{i}u")
                cons.append(con)
        return cons



    def create_variables(self):
        # H (height), d (diameter), t (thickness), B (seperation distance), E (modulus of elasticity), p (density)
        var_names = ["H", "d", "t", "B", "E", "p"]

        self.initial_values = np.array([30.0, 3.0, 0.1, 60.0, 30000., 0.3])
        lower_bounds = np.array([20.0, 0.5, 0.01, 20.0, 25000., 0.01])
        upper_bounds = np.array([60.0, 5.0, 1.0, 100.0, 40000., 0.5])

        # Create a list of Variables for MOProblem class
        variables = variable_builder(var_names, self.initial_values, lower_bounds, upper_bounds)
        return variables

    def create_problem(self):
        Problem = MOProblem(objectives=self.objectives, variables=self.variables, constraints=self.constraints)
        scipy_de_method = ScalarMethod(
            lambda x, _, **y: minimize(x, **y, x0 = self.initial_values),
            method_args={"method":"SLSQP"},
            use_scipy=True
        )
        ideal, nadir = payoff_table_method(Problem, initial_guess=self.initial_values, solver_method=scipy_de_method)
        print(f"Nadir: {nadir}\nIdeal: {ideal}")

        Problem.ideal = ideal
        Problem.nadir = nadir

        return Problem, scipy_de_method

#*******************************************************************************

    def optimize(self):
        step_sizes = np.array([5, 55, 5, 4])[self.obj]

        var, obj = solve_pareto_front_representation(self.problem, step_sizes)

        data = np.concatenate((obj,var), axis=1)

        return data

    def add_labels(self, data, RandomData, columns, filename='labelled_TBT.csv'):
        new_data = []
        for row in data:
            row = row.tolist()
            if row[0] <= 10 and row[1] <= 100 and row[2] <= 10:                     #KP
                row.extend([0, "KP"])
                new_data.append(row)
            elif row[0] <= 10 and row[1] > 100 and row[2] > 10:                     #F1
                row.extend([1, "F1"])
                new_data.append(row)
            elif row[0] > 10 and row[1] <= 100 and row[2] > 10:                     #F2
                row.extend([2, "F2"])
                new_data.append(row)
            elif row[0] > 10 and row[1] > 100 and row[2] <= 10:                     #F3
                row.extend([3, "F3"])
                new_data.append(row)
            elif row[0] <= 10 and row[1] <= 100 and row[2] > 10:                     #F12
                row.extend([4, "F12"])
                new_data.append(row)
            elif row[0] > 10 and row[1] <= 100 and row[2] <= 10:     #F23
                row.extend([5, "F23"])
                new_data.append(row)
            elif row[0] <= 10 and row[1] > 100 and row[2] <= 10:   #F13
                row.extend([6, "F13"])
                new_data.append(row)
        for row in RandomData:
            row = row.tolist()
            if row[0] > 10 and  row[1] > 100  and row[2] > 10:                       #BD
                row.extend([7, "BD"])
                new_data.append(row)
        df = pd.DataFrame(new_data,columns=columns)
        return df
