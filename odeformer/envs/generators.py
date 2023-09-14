# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from ast import parse
from operator import length_hint, xor
from typing import Union
# from turtle import degrees
import numpy as np
import scipy.special
import copy
from logging import getLogger
from collections import defaultdict
import warnings
import odeformer
from odeformer.envs import encoders
from odeformer.envs.utils import *
from ..utils import bool_flag, timeout, MyTimeoutError
from functools import partial
import numexpr as ne

# import numba as nb
# from numba import njit
# from numbalsoda import lsoda_sig, lsoda
# import nbkode

#from odeformer.envs.export_jax import *
# import jax
# import jax.numpy as jnp
# from diffrax import diffeqsolve, ODETerm, SaveAt
# from diffrax.solver import *

# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from diffeqpy import ode

warnings.filterwarnings("ignore")
import traceback

logger = getLogger()
import random

operators_real = {
    "add"   : 2,
    "sub"   : 2,
    "mul"   : 2,
    "div"   : 2,
    "abs"   : 1,
    "inv"   : 1,
    "sqrt"  : 1,
    "log"   : 1,
    "exp"   : 1,
    "sin"   : 1,
    "arcsin": 1,
    "cos"   : 1,
    "arccos": 1,
    "tan"   : 1,
    "arctan": 1,
    "pow2"  : 1,
    "pow3"  : 1,
    'id'    : 1
}

operators_extra = {"pow": 2}

math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]
all_operators = {**operators_real, **operators_extra}


class Node:
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params

    def push_child(self, child):
        self.children.append(child)

    def prefix(self, skeleton=False):
        s = str(self.value)
        if skeleton:
            try: 
                float(s)
                s = "CONSTANT"
            except:
                pass
        for c in self.children:
            s += "," + c.prefix(skeleton=skeleton)
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self, skeleton=False):
        s = str(self.value)
        if skeleton:
            try: 
                float(s)
                s = "CONSTANT"
            except:
                pass
        nb_children = len(self.children)
        if nb_children == 0:
            return s
        if nb_children == 1:
            if s == "pow2":
                s = "(" + self.children[0].infix(skeleton=skeleton) + ")**2"
            elif s == "inv":
                s = "1/(" + self.children[0].infix(skeleton=skeleton) + ")"
            elif s == "pow3":
                s = "(" + self.children[0].infix(skeleton=skeleton) + ")**3"
            else:
                s = s + "(" + self.children[0].infix(skeleton=skeleton) + ")"
            return s
        else:
            if s == "add":
                return self.children[0].infix(skeleton=skeleton) + " + " + self.children[1].infix(skeleton=skeleton)
            if s == "sub":
                return self.children[0].infix(skeleton=skeleton) + " - " + self.children[1].infix(skeleton=skeleton)
            if s == "pow":
                res  = "(" + self.children[0].infix(skeleton=skeleton) + ")**"
                res += ("" + self.children[1].infix(skeleton=skeleton))
                return res
            elif s == "mul":
                res  = "(" + self.children[0].infix(skeleton=skeleton) + ")" if self.children[0].value in ["add","sub"] else (self.children[0].infix(skeleton=skeleton))
                res += " * "
                res += "(" + self.children[1].infix(skeleton=skeleton) + ")" if self.children[1].value in ["add","sub"] else (self.children[1].infix(skeleton=skeleton))
                return res
            elif s == "div":
                res  = "(" + self.children[0].infix(skeleton=skeleton) + ")" if self.children[0].value in ["add","sub"] else (self.children[0].infix(skeleton=skeleton))
                res += " / "
                res += "(" + self.children[1].infix(skeleton=skeleton) + ")" if self.children[1].value in ["add","sub"] else (self.children[1].infix(skeleton=skeleton))
                return res


    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        # infix a default print
        return self.prefix()

    def __repr__(self):
        # infix a default print
        return str(self)
    
    def val(self, x, t, deterministic=True):
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim]
            elif str(self.value) == "t":
                return t
            elif str(self.value) == "rand":
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))

        if self.value == "add":
            return self.children[0].val(x,t) + self.children[1].val(x,t)
        if self.value == "sub":
            return self.children[0].val(x,t) - self.children[1].val(x,t)
        if self.value == "mul":
            m1, m2 = self.children[0].val(x,t), self.children[1].val(x,t)
            try:
                return m1 * m2
            except Exception as e:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x,t), self.children[1].val(x,t)
            try:
                return np.power(m1, m2)
            except Exception as e:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(self.children[0].val(x,t), self.children[1].val(x,t))
        if self.value == "min":
            return np.minimum(self.children[0].val(x,t), self.children[1].val(x,t))

        if self.value == "div":
            denominator = self.children[1].val(x,t)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x,t) / denominator
            except Exception as e:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "inv":
            denominator = self.children[0].val(x,t)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception as e:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x,t)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "sqrt":
            numerator = self.children[0].val(x,t)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x,t)
            try:
                return numerator ** 2
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x,t)
            try:
                return numerator ** 3
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x,t))
        if self.value == "sign":
            return (self.children[0].val(x,t) >= 0) * 2.0 - 1.0
        if self.value == "step":
            x = self.children[0].val(x,t)
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x,t)
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x,t))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x,t))[
                0
            ]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x,t))
                except Exception as e:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x,t))
            assert False, "Could not find function"

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)


class NodeList:
    def __init__(self, nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self, skeleton=False):
        return " | ".join([node.infix(skeleton=skeleton) for node in self.nodes])

    def __len__(self):
        return sum([len(node) for node in self.nodes])

    def prefix(self, skeleton=False):
        return ",|,".join([node.prefix(skeleton=skeleton) for node in self.nodes])

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, xs, t, deterministic=True):
        batch_vals = [
            np.expand_dims(node.val(np.copy(xs), t, deterministic=deterministic), -1)
            for node in self.nodes
        ]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value, new_value):
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)

    def get_dimension(self):
        return len(self.nodes)


class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate_datapoints(self, rng):
        pass


class RandomFunctions(Generator):
    def __init__(self, params, special_words):
        super().__init__(params)
        self.params = params
        self.prob_const = params.prob_const
        self.prob_rand = params.prob_rand
        self.prob_t = params.prob_t
        self.max_int = params.max_int
        self.min_binary_ops_per_dim = params.min_binary_ops_per_dim
        self.max_binary_ops_per_dim = params.max_binary_ops_per_dim
        self.min_unary_ops_per_dim = params.min_unary_ops_per_dim
        self.max_unary_ops_per_dim = params.max_unary_ops_per_dim
        self.min_dimension = params.min_dimension
        self.max_dimension = params.max_dimension
        self.max_number = 10 ** (params.max_exponent + params.float_precision)
        self.operators = copy.deepcopy(operators_real)

        self.operators_downsample_ratio = defaultdict(float)
        for operator in self.params.operators_to_use.split(","):
            operator, ratio = operator.split(":")
            ratio = float(ratio)
            self.operators_downsample_ratio[operator] = ratio

        if params.required_operators != "":
            self.required_operators = self.params.required_operators.split(",")
        else:
            self.required_operators = []

        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []

        self.unaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 1
        ] + self.extra_unary_operators

        self.binaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 2
        ] + self.extra_binary_operators

        unaries_probabilities = []
        for op in self.unaries:
            if op not in self.operators_downsample_ratio:
                unaries_probabilities.append(0.0)
            else:
                ratio = self.operators_downsample_ratio[op]
                unaries_probabilities.append(ratio)
        self.unaries_probabilities = np.array(unaries_probabilities)
        if self.unaries_probabilities.sum()==0:
            self.use_unaries = False
        else:
            self.unaries_probabilities /= self.unaries_probabilities.sum()
            self.use_unaries = True

        binaries_probabilities = []
        for op in self.binaries:
            if op not in self.operators_downsample_ratio:
                binaries_probabilities.append(0.0)
            else:
                ratio = self.operators_downsample_ratio[op]
                binaries_probabilities.append(ratio)
        self.binaries_probabilities = np.array(binaries_probabilities)
        self.binaries_probabilities /= self.binaries_probabilities.sum()

        self.unary = False  # len(self.unaries) > 0
        self.distrib = self.generate_dist(
            2 * self.max_binary_ops_per_dim * self.max_dimension
        )

        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants
        self.variables = ["rand"] + [f"x_{i}" for i in range(self.max_dimension)] + ["t"]
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )
        self.constants.remove("CONSTANT")

        if self.params.extra_constants is not None:
            self.extra_constants = self.params.extra_constants.split(",")
        else:
            self.extra_constants = []

        self.general_encoder = encoders.GeneralEncoder(
            params, self.symbols, all_operators
        )
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words

    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(
            len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1)
        ), "issue in generate_dist"
        return D

    def generate_float(self, rng, exponent=None):
        sign = rng.choice([-1, 1])
        # sample number loguniformly in max_prefactor, 1/max_prefactor
        constant = rng.uniform(np.log10(1/self.params.max_prefactor), np.log10(self.params.max_prefactor))
        constant = sign*10**constant
        return str(constant)

    def generate_int(self, rng):
        return str(rng.choice(self.constants + self.extra_constants))

    def generate_leaf(self, rng, dimension):
        if rng.rand() < self.prob_rand:
            return "rand"
        elif rng.rand() < self.prob_t:
            return "t"
        else:
            # if self.n_used_dims < dimension:
            #     dimension = self.n_used_dims
            #     self.n_used_dims += 1
            #     return f"x_{dimension}"
            # else:
            draw = rng.rand()
            if draw < self.prob_const:
                return self.generate_int(rng)
            else:
                dimension = rng.randint(0, dimension)
                return f"x_{dimension}"

    def generate_ops(self, rng, arity):
        if arity == 1:
            ops = self.unaries
            probas = self.unaries_probabilities
        else:
            ops = self.binaries
            probas = self.binaries_probabilities
        return rng.choice(ops, p=probas)

    def sample_next_pos(self, rng, nb_empty, nb_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    def generate_tree(self, rng, nb_binary_ops, dimension):
        self.n_used_dims = 0
        tree = Node(0, self.params)
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1
        while nb_binary_ops > 0:
            next_pos, arity = self.sample_next_pos(rng, nb_empty, nb_binary_ops)
            next_en += next_pos
            op = self.generate_ops(rng, arity)
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.params)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            next_en += 1
            nb_empty += arity - 1 - next_pos
            nb_binary_ops -= 1
        rng.shuffle(empty_nodes)
        for n in empty_nodes:
            if len(n.children) == 0:
                n.value = self.generate_leaf(rng, dimension)
        return tree

    def generate_multi_dimensional_tree(
        self,
        rng,
        dimension=None,
        nb_unary_ops=None,
        nb_binary_ops=None,
    ):
        trees = []

        if dimension is None:
            dimension = rng.randint(
                self.min_dimension, self.max_dimension + 1
            )

        if nb_binary_ops is None:
            nb_binary_ops_to_use = [
                rng.randint(
                    self.min_binary_ops_per_dim, self.max_binary_ops_per_dim+1
                )
                for dim in range(dimension)
            ]
        elif isinstance(nb_binary_ops, int):
            nb_binary_ops_to_use = [nb_binary_ops for _ in range(dimension)]
        else:
            nb_binary_ops_to_use = nb_binary_ops
        if nb_unary_ops is None:
            nb_unary_ops_to_use = [
                rng.randint(self.min_unary_ops_per_dim, self.max_unary_ops_per_dim + 1)
                for dim in range(dimension)
            ]
        elif isinstance(nb_unary_ops, int):
            nb_unary_ops_to_use = [nb_unary_ops for _ in range(dimension)]
        else:
            nb_unary_ops_to_use = nb_unary_ops

        for i in range(dimension):
            tree = self.generate_tree(rng, nb_binary_ops_to_use[i], dimension)
            if self.use_unaries:
                tree = self.add_unaries(rng, tree, nb_unary_ops_to_use[i])
            if self.params.reduce_num_constants:
                tree = self.add_prefactors(rng, tree)
            else:
                tree = self.add_linear_transformations(rng, tree, target=self.variables)
                tree = self.add_linear_transformations(rng, tree, target=self.unaries)
            trees.append(tree)
        tree = NodeList(trees)

        nb_unary_ops_to_use = [
            len([x for x in tree_i.prefix().split(",") if x in self.unaries])
            for tree_i in tree.nodes
        ]
        nb_binary_ops_to_use = [
            len([x for x in tree_i.prefix().split(",") if x in self.binaries])
            for tree_i in tree.nodes
        ]

        for op in self.required_operators:
            if op not in tree.infix():
                return self.generate_multi_dimensional_tree(
                    rng, dimension, nb_unary_ops, nb_binary_ops
                )

        return (
            tree,
            dimension,
            nb_unary_ops_to_use,
            nb_binary_ops_to_use,
        )

    def add_unaries(self, rng, tree, nb_unaries):
        prefix = self._add_unaries(rng, tree)
        prefix = prefix.split(",")
        indices = []
        for i, x in enumerate(prefix):
            if x in self.unaries:
                indices.append(i)
        rng.shuffle(indices)
        if len(indices) > nb_unaries:
            to_remove = indices[: len(indices) - nb_unaries]
            for index in sorted(to_remove, reverse=True):
                del prefix[index]
        tree = self.equation_encoder.decode(prefix).nodes[0]
        return tree

    def _add_unaries(self, rng, tree):

        s = str(tree.value)

        for c in tree.children:
            if len(c.prefix().split(",")) < self.params.max_unary_depth:
                unary = rng.choice(self.unaries, p=self.unaries_probabilities)
                if unary=='id': s += "," + self._add_unaries(rng, c)
                s += f",{unary}," + self._add_unaries(rng, c)
            else:
                s += f"," + self._add_unaries(rng, c)
        return s

    def add_prefactors(self, rng, tree):
        transformed_prefix = self._add_prefactors(rng, tree)
        if transformed_prefix == tree.prefix():
            a = self.generate_float(rng) if rng.rand() < self.params.prob_prefactor else transformed_prefix
            transformed_prefix = f"mul,{a}," + transformed_prefix
        #a = self.generate_float(rng)
        #transformed_prefix = f"add,{a}," + transformed_prefix if rng.rand() < self.params.prob_prefactor else transformed_prefix
        tree = self.equation_encoder.decode(transformed_prefix.split(",")).nodes[0]
        return tree

    def _add_prefactors(self, rng, tree):

        s = str(tree.value)
        a, b, c = [self.generate_float(rng) for _ in range(3)]
        add_prefactor = f",add,{a}," if rng.rand() < self.params.prob_prefactor else ","
        mul_prefactor1 = f",mul,{b}," if rng.rand() < self.params.prob_prefactor else ","
        mul_prefactor2 = f",mul,{c}," if rng.rand() < self.params.prob_prefactor else ","
        total_prefactor = add_prefactor.rstrip(",") + "," + mul_prefactor1.lstrip(",")
        if s in ["add", "sub"]:
            s += (
                "," if tree.children[0].value in ["add", "sub"] else mul_prefactor1
            ) + self._add_prefactors(rng, tree.children[0])
            s += (
                "," if tree.children[1].value in ["add", "sub"] else mul_prefactor2
            ) + self._add_prefactors(rng, tree.children[1])
        elif s in self.unaries and tree.children[0].value not in ["add", "sub"]:
            s += total_prefactor + self._add_prefactors(rng, tree.children[0])
        else:
            for c in tree.children:
                s += f"," + self._add_prefactors(rng, c)
        return s

    def add_linear_transformations(self, rng, tree, target, add_after=False):

        prefix = tree.prefix().split(",")
        indices = []

        for i, x in enumerate(prefix):
            if x in target:
                indices.append(i)

        offset = 0
        for idx in indices:
            a, b = self.generate_float(rng), self.generate_float(rng)
            if add_after:
                prefix = (
                    prefix[: idx + offset + 1]
                    + ["add", a, "mul", b]
                    + prefix[idx + offset + 1 :]
                )
            else:
                prefix = (
                    prefix[: idx + offset]
                    + ["add", a, "mul", b]
                    + prefix[idx + offset :]
                )
            offset += 4
        tree = self.equation_encoder.decode(prefix).nodes[0]

        return tree

    def relabel_variables(self, tree):
        active_variables = []
        for elem in tree.prefix().split(","):
            if elem.startswith("x_"):
                active_variables.append(elem)
        active_variables = list(set(active_variables))
        dimension = len(active_variables)
        if dimension == 0:
            return 0
        active_variables.sort(key=lambda x: int(x[2:]))
        for j, xi in enumerate(active_variables):
            tree.replace_node_value(xi, "x_{}".format(j))
        return dimension

    def function_to_skeleton(
        self, tree, skeletonize_integers=False, constants_with_idx=False
    ):
        constants = []
        prefix = tree.prefix().split(",")
        j = 0
        for i, pre in enumerate(prefix):
            try:
                float(pre)
                is_float = True
                if pre.lstrip("-").isdigit():
                    is_float = False
            except ValueError:
                is_float = False

            if pre.startswith("CONSTANT"):
                constants.append("CONSTANT")
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                j += 1
            elif is_float or (pre is self.constants and skeletonize_integers):
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                else:
                    prefix[i] = "CONSTANT"
                while i > 0 and prefix[i - 1] in self.unaries:
                    del prefix[i - 1]
                try:
                    value = float(pre)
                except:
                    value = getattr(np, pre)
                constants.append(value)
                j += 1
            else:
                continue

        new_tree = self.equation_encoder.decode(prefix)
        return new_tree, constants

    def wrap_equation_floats(self, tree, constants):
        tree = self.tree
        env = self.env
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(
            tree, constants
        )
        tree_with_constants = env.word_to_infix(prefix, is_float=False, str_array=False)
        return tree_with_constants

    def generate_datapoints(
        self,
        tree,
        rng,
        dimension,
        n_points,
        n_init_conditions=1
        ):

        if self.params.fixed_init_scale:
            y0 = np.ones(dimension)
        else:
            y0 = self.params.init_scale * rng.randn(dimension)
        times = np.linspace(1, self.params.time_range, n_points)
        #times = times.repeat(n_init_conditions, axis=0)

        stop_value = self.params.max_trajectory_value
        def stop_event(t, y):
            return np.min(stop_value-abs(y))
        stop_event.terminal = True

        trajectory = integrate_ode(y0, times, tree, self.params.ode_integrator, events=stop_event, debug=self.params.debug)

        if trajectory is None:
            return None, None
        if np.any(np.isnan(trajectory)):
            return None, None
        if np.any(np.abs(trajectory)>10**self.params.max_exponent):
            return None, None
        if rng.rand() < self.params.discard_stationary_trajectory_prob:
            window_len = n_points//4
            last = trajectory[-window_len:]
            if np.all(np.abs((np.max(last, axis=0)-np.min(last, axis=0))/window_len) < 1e-3): # remove constant
                return None, None
        
        return tree, (times, trajectory)

@timeout(1)
def _integrate_ode(y0, times, tree, ode_integrator = 'solve_ivp', events=None, debug=False, allow_warnings=False):

    with warnings.catch_warnings(record=True) as caught_warnings:

        if ode_integrator == 'jax':

            sympy_trees = odeformer.envs.Simplifier.tree_to_sympy_expr(tree, round=True)
            jax_trees, jax_params = [], []
            for tree in sympy_trees:
                symbols = list(tree.free_symbols)
                jax_tree, jax_param = sympy2jax(tree, symbols)
                jax_trees.append(jax_tree); jax_params.append(jax_param)
        else:
            tree = tree_to_numexpr_fn(tree)

        if ode_integrator == 'jax':
            def func(t,y,args=None):
                return jnp.concatenate([jax_tree(y, jax_param).reshape(-1,1) for jax_tree, jax_param in zip(jax_trees, jax_params)], axis=1)
            term = ODETerm(func)
            solver = Euler()
            t0 = min(times)  # start of integration
            t1 = max(times)  # end of integration
            dt0 = 0.1  # Initial stepsize (solver is then adaptive)
            saveat = SaveAt(ts=times) # want regular output at 256 points

            # convenience wrapper to only supply y0 as input to solver
            p_diffeqsolve = lambda init : diffeqsolve(term, solver, t0, t1, dt0, init, saveat=saveat)
            if len(y0.shape)<2: y0 = y0.reshape(1,-1)
            y0 = jnp.array(y0)
            sol = p_diffeqsolve(y0)
            trajectory = sol.ys
            trajectory = trajectory.reshape(trajectory.shape[0], trajectory.shape[2])

        elif ode_integrator == 'numba':
            def func(u,p,t):
                derivs = tree([u],[t])[0]  
            #func = numba.jit(func)
            try: 
                sol = ode.ODEProblem(func, y0, (min(times), max(times))).solve()
                trajectory = sol(times)
            except: 
                if debug:
                    print(traceback.format_exc())
                return None    
            
        elif ode_integrator == "nbkode":
            @njit
            def func(y,t):
                return tree([y],[t])[0]
            solver = nbkode.ForwardEuler(func, min(times), y0)
            trajectory = solver.run(times)

        elif ode_integrator == "odeint":
            #@njit
            def func(y,t):
                return tree([y],[t])[0]
            with stdout_redirected():
                try: 
                    trajectory = scipy.integrate.odeint(func, y0, times)#, rtol=1e-2, atol=1e-6)
                    if abs(trajectory[-10:].max()) < 1e-100:
                        return None
                except: 
                    if debug:
                        print(traceback.format_exc())
                    return None
                    
        elif ode_integrator == "solve_ivp":
            #@njit
            times = np.array(times)+1
            def func(t,y):
                ret = tree([y],[t])
                return ret[0]
            try: 
                trajectory = scipy.integrate.solve_ivp(func, (min(times), max(times)), y0, t_eval=times, events=events)#, method='RK23', rtol=1e-2, atol=1e-6)
                events = trajectory.t_events
                trajectory = trajectory.y.T
            except: 
                if debug:
                    print(traceback.format_exc())
                return None
                
        else:
            raise NotImplementedError
    
    #if debug: print(trajectory, np.any(np.isnan(trajectory)), len(times)!=len(trajectory), len(times), len(trajectory))#, solved_times)
    
    if np.any(np.isnan(trajectory)) or len(times)!=len(trajectory):
        if debug: print("bad integration")
        return [np.nan for _ in range(len(times))]
    if len(caught_warnings) > 0 and not allow_warnings:
        if debug: print("Caught warning")
        return [np.nan for _ in range(len(times))]
    
    return trajectory

def integrate_ode(y0, times, tree, ode_integrator = 'solve_ivp', events=None, debug=False, allow_warnings=False):
    try: 
        return _integrate_ode(y0, times, tree, ode_integrator, events, debug, allow_warnings)
    except MyTimeoutError:
        if debug: print("Timeout error")
        return [np.nan for _ in range(len(times))]

def tree_to_numexpr_fn(tree):
    if not isinstance(tree, str):
        infix = tree.infix()
    else:
        infix = tree
        
    numexpr_equivalence = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "pow": "**",
        "inv": "1/",
        " | ": "|"
    }

    for old, new in numexpr_equivalence.items():
        infix = infix.replace(old, new)

    #@njit
    def wrapped_numexpr_fn(x, t, extra_local_dict={}):
        #t, x = np.array(t), np.array(x)
        local_dict = {}
        dimension = len(x[0])
        for d in range(dimension): 
            local_dict["x_{}".format(d)] = np.array(x)[:, d]

        local_dict["t"] = t[:]
        local_dict.update(extra_local_dict)
        # predicted_dim = len(infix.split('|'))
        try:
            if '|' in infix:    
                vals = np.concatenate([ne.evaluate(node, local_dict=local_dict).reshape(-1,1) for node in infix.split('|')], axis=1)
            else:
                vals = ne.evaluate(infix, local_dict=local_dict).reshape(-1,1)
        except Exception as e:
            # print(e)
            # print("problem with tree", infix)
            # print(traceback.format_exc())
            vals = np.array([np.nan for _ in range(x.shape[0])])#.reshape(-1, 1).repeat(predicted_dim, axis=1)
        return vals

    return wrapped_numexpr_fn



if __name__ == "__main__":

    from parsers import get_parser
    from odeformer.envs.environment import SPECIAL_WORDS

    parser = get_parser()
    params = parser.parse_args()
    generator = RandomFunctions(params, SPECIAL_WORDS)
    rng = np.random.RandomState(0)
    tree, _, _, _, _ = generator.generate_multi_dimensional_tree(
        np.random.RandomState(0), dimension=1
    )
    print(tree)
    x, y = generator.generate_datapoints(rng, tree, "gaussian", 10, 200, 200)
    generator.order_datapoints(x, y)
