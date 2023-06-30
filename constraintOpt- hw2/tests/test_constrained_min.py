import unittest

from src import utils
from src.constrained_min import interior_pt
from tests import examples
import numpy as np


class TestMinimizer(unittest.TestCase):

    def test_qp(self):
        f = examples.Function1()
        x0 = [0.1, 0.2, 0.7]
        x_list,objectives = interior_pt(f.func, f.ineq_constraints, f.eq_constraints_matrix, f.eq_constraints_res, x0)
        x = x_list[-1]
        inq_constraint = []
        for i in f.ineq_constraints:
            res = i(x)[0]
            inq_constraint.append(res)
        print("quadratic programming")
        print(" * final point is: ", x)
        print(" * final objective is: ", f.func(x)[0])
        print(" * final inequality constraints: ", inq_constraint)
        print(" * final equality constraints: ", np.dot(f.eq_constraints_matrix, x))

        utils.plot_result_3d(x_list,angle=(5,30))
        utils.plot_result_3d(x_list,angle=(50,45))
        utils.plot_value_iterations(objectives, name_of_function="quadratic programming")

    def test_lp(self):
        f = examples.Function2()
        x0 = [0.5, 0.75]
        x_list, objectives = interior_pt(f.func, f.ineq_constraints, f.eq_constraints_matrix, f.eq_constraints_res, x0)
        x = x_list[-1]
        inq_constraint = []
        for i in f.ineq_constraints:
            res = i(x)[0]
            inq_constraint.append(res)
        print("linear programming")
        print(" * final point is: ", x)
        print(" * final objective is: ", f.func(x)[0])
        print(" * final inequality constraints: ", inq_constraint)
        print(" * final equality constraints: ", None)
        utils.plot_result(x_list)
        utils.plot_value_iterations(-1*np.array(objectives), name_of_function="linear programing")
