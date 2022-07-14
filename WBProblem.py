from MOOCourse.modules.TwoBarTruss.problem import create_problem
from MOOCourse.modules.utils import save

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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

class WBProblem():
    def __init__(self):
            self.load = 66
            # weight, stress, buckling stress and deflection
            self.obj = np.array([True, True])
            self.initial_values = None
            self.constraints = self.create_constraints()
            self.objectives = self.create_objectives()
            self.variables = self.create_variables()
            self.problem,self.method = self.create_problem()


    def create_constraints(self):

        parameters = [30000000,12000000,14,13600,30000,0.25,6000,0.10471,0.04811,1]

        def con1(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            E,G,L,t_max,s_max,d_max,P,C1,C2,C3 = parameters
            t_d = P/(np.sqrt(2)*x1*x2)
            M = P * (L + x2/2)
            R = np.sqrt(pow(x2,2)/4 + pow((x1+x3)/2,2))
            J = 2*(x1*x2*np.sqrt(2)*(pow(x2,2)/12 + (pow((x1+x3)/2,2))))
            t_dd = (M * R) / J
            t = np.sqrt(pow(t_d,2) + 2*t_d*t_dd*(x2/(2*R))+pow(t_dd,2))
            con1_value = t_max - t
            return con1_value

        def con2(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            E,G,L,t_max,s_max,d_max,P,C1,C2,C3 = parameters
            s = (6*P*L)/(x4* pow(x3,2))
            con2_value = s_max - s
            return con2_value

        def con3(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            con3_value = x4-x1
            return con3_value

        def con4(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            E,G,L,t_max,s_max,d_max,P,C1,C2,C3 = parameters
            con4_value = -1 * C1*pow(x1,2)+C2*x3*x4*(L+x2)-5
            return con4_value

        def con5(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            con5_value = x1 - 0.125
            return con5_value

        def con6(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            E,G,L,t_max,s_max,d_max,P,C1,C2,C3 = parameters
            d = (4*P*pow(L,3))/(E*x4*pow(x3,2))
            con6_value = d_max - d
            return con6_value

        def con7(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            E,G,L,t_max,s_max,d_max,P,C1,C2,C3 = parameters
            Pc = ((4.013*E*np.sqrt((pow(x3,2)*pow(x4,6))/36))/pow(L,2))*(1-(x3/(2*L))*np.sqrt(E/(4*G)))
            con7_value = Pc - P
            return con7_value

        def con8(xs: np.ndarray, _) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T
            A = x1 - 0.1
            B = x2 - 0.1
            C = 10.0 - x3
            D = 2.0 - x4
            if np.array([A,B,C,D]).any() < 0:
                return -100
            return 100

        constraints = [
            ScalarConstraint("con1", 4, 2, con1),
            ScalarConstraint("con2", 4, 2, con2),
            ScalarConstraint("con3", 4, 2, con3),
            ScalarConstraint("con4", 4, 2, con4),
            ScalarConstraint("con5", 4, 2, con5),
            ScalarConstraint("con6", 4, 2, con6),
            ScalarConstraint("con7", 4, 2, con7),
            ScalarConstraint("con8", 4, 2, con8),
        ]
        return constraints

    def create_objectives(self, obj_mask = [True]*2):

        if type(obj_mask) is not np.ndarray:
            obj_mask = np.array(obj_mask)

        # Defining the objective functions
        def V_weld(xs: np.ndarray) -> np.ndarray:
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T  # Assign the values to named variables for clarity
            return pow(x1,2)*x2

        def V_bar(xs: np.ndarray) -> np.ndarray:
            L = 14 # parameter
            xs = np.atleast_2d(xs)
            x1,x2,x3,x4 = xs.T  # Assign the values to named variables for clarity
            return x3 * x4 * (L+x2)

        # Define objectives
        obj1 = _ScalarObjective("V_weld", V_weld)
        obj2 = _ScalarObjective("V_bar", V_bar)
        objectives = np.array([obj1, obj2])[obj_mask]

        return objectives

    def create_variables(self):
        # x1,x2 weld dimensions | x3,x4 beam dimensions
        var_names = ["x1","x2","x3","x4"]

        self.initial_values = np.array([1.0, 1.0, 1.0, 5.0])
        lower_bounds = np.array([0.0, 0.0, 0.0, 0.0])
        upper_bounds = np.array([10.0, 10.0, 10.0, 10.0])

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

    def optimize(self):
        step_sizes = np.array([5.0, 5.0])[self.obj]

        var, obj = solve_pareto_front_representation(self.problem, step_sizes)

        data = np.concatenate((obj,var), axis=1)

        return data

    def add_labels(self,data,columns):
        new_data = []
        for row in data:
            row = row.tolist()
            if row[0] > 0 and row[0] < 1 and row[1] > 0 and row[1] < 40: #KP
                row.extend([0, "KP"])
                new_data.append(row)
            elif row[0] > 3 and row[0] < 5 and row[1] > 0 and row[1] < 40: #F1
                row.extend([1, "F1"])
                new_data.append(row)
            elif row[0] > 0 and row[0] < 1 and row[1] > 60 and row[1] < 100: #F2
                row.extend([2, "F2"])
                new_data.append(row)
            elif row[0] > 3 and row[0] < 5 and row[1] > 60 and row[1] < 100: #BD
                row.extend([3, "BD"])
                new_data.append(row)
        df = pd.DataFrame(new_data,columns=columns)
        return df
