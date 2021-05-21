from typing import List

import autograd.numpy as anp
import tensorflow as tf
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

from trieste.models import ModelStack
from trieste.type import TensorType

from .parametric_gp_posterior import gen_approx_posterior_through_rff_wsa


def find_pareto_front_from_sampled_gp_posterior(
    gp_model: ModelStack,
    sample_pf_num: int,
    search_space,
    popsize: int = 20,
    num_moo_iter: int = 300,
) -> List[TensorType]:
    """
    :param gp_model
    :param sample_pf_num
    :param search_space
    :param popsize
    :param num_moo_iter
    """
    gp_post_samples = []
    for obj_model in gp_model._models:  # gen rff_wsa sample for each obj
        gp_post_samples.append(gen_approx_posterior_through_rff_wsa(obj_model, sample_pf_num))

    pf_samples = []
    for i in range(sample_pf_num):  # calculate pf
        obj_model = lambda x: tf.concat(
            [obj_post_sample[i](x) for obj_post_sample in gp_post_samples], axis=1
        )
        pf_samples.append(
            moo_optimize(
                obj_model,
                len(gp_model._models),
                len(search_space.lower),
                (search_space.lower, search_space.upper),
                popsize,
                num_generation=num_moo_iter,
            )
        )
    return pf_samples


def moo_optimize(f, input_dim: int, obj_num: int, bounds: tuple, popsize: int, num_generation: int):
    """
    :param f
    :param input_dim
    :param bounds
    :param popsize
    :param num_generation
    """

    class MyProblem(Problem):
        def __init__(self, n_var, n_obj, n_constr: int = 0):
            super().__init__(
                n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=bounds[0], xu=bounds[1]
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = f(x)

    problem = MyProblem(n_var=input_dim, n_obj=obj_num)
    algorithm = NSGA2(
        pop_size=popsize,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", num_generation), save_history=False, verbose=False)
    return res
