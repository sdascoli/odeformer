# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Union
from argparse import Namespace
import numpy as np
import math
import sympy as sp
from .generators import Node, NodeList
from .utils import *

class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """

    def __init__(self, params):
        pass

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def decode(self, lst):
        pass


class GeneralEncoder:
    def __init__(self, params, symbols, all_operators):
        self.constant_encoder = ConstantEncoder(params) if params.use_two_hot else None
        self.float_encoder = FloatSequences(params)
        self.prefix_encoder = Equation(params, symbols, self.float_encoder, all_operators, self.constant_encoder)
        self.infix_encoder = InfixEncoder(params, symbols, self.float_encoder, all_operators, self.constant_encoder)
        if params.use_infix:
            self.equation_encoder = self.infix_encoder
        else:
            self.equation_encoder = self.prefix_encoder

class ConstantEncoder(Encoder):
    
    def __init__(self, params: Namespace, force_min_max: List[int]=None):
            
        # TODO: do we want / need to remove int tokens from the equation encoder vocabulary?
        
        if force_min_max is not None:
            assert force_min_max[0] < force_min_max[1], \
                f"`force_min_max[0]` should be smaller than `force_min_max[1]` but `force_min_max` is {force_min_max}"
            self.min = force_min_max[0]
            self.max = force_min_max[1]
        else:
            # See RandomFunctions.generate_float(...) in generators.py
            sign = 1
            mantissa = float(10 ** params.float_precision)
            max_power = (params.max_exponent_prefactor - (params.float_precision + 1) // 2)
            exponent = max_power
            constant = int(np.ceil(sign * (mantissa * 10 ** exponent)))
            self.min = -constant
            self.max = constant
            assert (-params.max_int) >= self.min, \
                f"params.min_int (= {params.min_int}) is smaller than supported constant range (self.min = {self.min})."
            assert params.max_int <= self.max, \
                f"params.max_int (= {params.max_int}) is larger than supported constant range (self.max = {self.max})."
        # required for compatibility
        self.symbols = [f"c{i}" for i in range(self.min, self.max+1)]

    def encode(self, values: Union[List, np.ndarray, int, float]) -> List[str]:
        if isinstance(values, (int, float)): # TODO: Are there other valid numeric types?
            values = [values]
        assert isinstance(values, List) or isinstance(values, np.ndarray), type(values)
        assert len(values) == 1, len(values)
        return [f"{values[0]:+}"]
    
    def decode(self, lst: Union[List, np.ndarray, str, float, int]) -> Union[List, np.ndarray]:
        if isinstance(lst, str) or isinstance(lst, float) or isinstance(int):
            lst = [lst]
        assert isinstance(lst, List) or isinstance(lst, np.ndarray), type(lst)
        assert len(lst) == 1, lst
        return lst

class FloatSequences(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.float_precision = params.float_precision
        self.max_exponent = params.max_exponent
        self.base = (self.float_precision + 1)
        self.max_token = 10 ** self.base
        self.symbols = []
        self.sign_as_token = params.sign_as_token
        if self.sign_as_token:
            self.symbols.extend(["SIGN+", "SIGN-"])
            self.symbols.extend(["N" + f"%0{self.base}d" % i for i in range(self.max_token)])
        else:
            self.symbols.extend(["+N" + f"%0{self.base}d" % i for i in range(self.max_token)])
            self.symbols.extend(["-N" + f"%0{self.base}d" % i for i in range(self.max_token)])
        self.symbols.extend(
            ["E" + str(i) for i in range(-self.max_exponent, self.max_exponent + 1)]
        )
        self.float_descriptor_length = 3 if self.sign_as_token else 2

    def encode(self, values):
        """
        Write a float number
        """
        precision = self.float_precision
        if isinstance(values, float):
            values = [values]
        values = np.array(values)
        if len(values.shape) == 1:
            seq = []
            for val in values:
                assert val not in [-np.inf, np.inf], "cannot encode infinity"
                sign = "+" if val >= 0 else "-"
                m, e = (f"%.{precision}e" % val).split("e")
                i, f = m.lstrip("-").split(".")
                mantissa = i + f
                expon = int(e) - precision
                if expon < -100:
                    mantissa = ["0" * base]
                    expon = int(0)
                if self.sign_as_token:
                    token_sequence = ["SIGN"+sign, f"N{mantissa}", f"E{expon}"]
                else:
                    token_sequence = [f"{sign}N{mantissa}",f"E{expon}"]
                seq += token_sequence
            return seq
        else:
            seqs = [self.encode(val) for val in values]
        return seqs

    def decode(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        # TODO: does it really return the position it ends in the list?
        if len(lst) == 0:
            return None
        seq = []
        for val in chunks(lst, self.float_descriptor_length):
            try:
                mant = ""
                if self.sign_as_token:
                    sign = 1 if val[0] == "SIGN+" else -1
                    for x in val[1:-1]:
                        mant += x[1:]                
                    exp = int(val[-1][1:])
                else:
                    sign = 1 if val[0][0] == "+" else -1
                    for x in val[:-1]:
                        mant += x[2:]  
                    exp = int(val[-1][1:])
                mant = int(mant)
                value = sign * float(f"{mant}e{exp}")
            except Exception:
                import traceback
                if self.params.debug:
                    print(traceback.format_exc())
                value = np.nan
            seq.append(value)
        return seq

class Equation(Encoder):
    def __init__(self, params, symbols, float_encoder, all_operators, constant_encoder=None):
        super().__init__(params)
        self.params = params
        self.max_int = self.params.max_int
        self.symbols = symbols
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []
        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        self.float_encoder = float_encoder
        self.all_operators = all_operators
        self.constant_encoder = constant_encoder
        self.float_descriptor_length = 3 if self.params.sign_as_token else 2

    def encode(self, tree):
        res = []
        for elem in tree.prefix().split(","):
            try:
                val = float(elem)
                if self.constant_encoder is not None:
                    res.extend(self.constant_encoder.encode(val))
                else:
                    if elem.lstrip("-").isdigit():
                        res.extend(self.write_int(int(elem)))
                    else:
                        res.extend(self.float_encoder.encode(np.array([val])))
            except ValueError:
                res.append(elem)
        return res

    def _decode(self, lst):
        if len(lst) == 0:
            return None, 0
        # elif (lst[0] not in self.symbols) and (not lst[0].lstrip("-").replace(".","").replace("e+", "").replace("e-","").isdigit()):
        #     return None, 0
        elif "OOD" in lst[0]:
            return None, 0
        elif lst[0] in self.all_operators.keys():
            res = Node(lst[0], self.params)
            arity = self.all_operators[lst[0]]
            pos = 1
            for i in range(arity):
                child, length = self._decode(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[0].startswith("INT"):
            val, length = self.parse_int(lst)
            return Node(str(val), self.params), length
        elif lst[0] == "SIGN+" or lst[0] == "SIGN-" or lst[0].startswith("+N") or lst[0].startswith('-N'):
            if self.params.use_two_hot:
                return Node(str(lst[0]), self.params), 1
            else:
                try:
                    val = self.float_encoder.decode(lst[:self.float_descriptor_length])[0]
                except Exception as e:
                    return None, 0
                return Node(str(val), self.params), self.float_descriptor_length
        elif (
            lst[0].startswith("CONSTANT") or lst[0] == "y"
        ):  ##added this manually CAREFUL!!
            return Node(lst[0], self.params), 1
        elif lst[0] in self.symbols:
            return Node(lst[0], self.params), 1
        else:
            try:
                float(lst[0])  # if number, return leaf
                return Node(lst[0], self.params), 1
            except:
                return None, 0

    def split_at_value(self, lst, value):
        indices = [i for i, x in enumerate(lst) if x == value]
        res = []
        for start, end in zip(
            [0, *[i + 1 for i in indices]], [*[i - 1 for i in indices], len(lst)]
        ):
            res.append(lst[start : end + 1])
        return res

    def decode(self, lst):
        trees = []
        lists = self.split_at_value(lst, "|")
        for lst in lists:
            tree = self._decode(lst)[0]
            if tree is None:
                return None
            trees.append(tree)
        tree = NodeList(trees)
        return tree

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.max_int
        val = 0
        i = 0
        for x in lst[1:]:
            if not (x.rstrip("-").isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == "INT-":
            val = -val
        return val, i + 1

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        """
        if not self.params.use_sympy:
            return [str(val)]

        base = self.max_int
        res = []
        max_digit = abs(base)
        neg = val < 0
        val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        res.append("INT-" if neg else "INT+")
        return res[::-1]


    
class InfixEncoder(Encoder):
    def __init__(self, params, symbols, float_encoder, all_operators, constant_encoder=None):
        super().__init__(params)
        self.params = params
        self.max_int = self.params.max_int
        self.symbols = symbols
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []
        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        self.float_encoder = float_encoder
        self.all_operators = all_operators
        self.constant_encoder = constant_encoder
        self.float_descriptor_length = 3 if self.params.sign_as_token else 2
        self.equation_encoder = Equation(params, symbols, float_encoder, all_operators, constant_encoder) # TODO: improve this

    def encode(self, tree):
        res = []
        for elem in tree.infix().split(" "):
            try:
                val = float(elem)
                if self.constant_encoder is not None:
                    res.extend(self.constant_encoder.encode(val))
                else:
                    if elem.lstrip("-").isdigit():
                        res.append(elem)
                    else:
                        res.extend(self.float_encoder.encode(np.array([val])))
            except ValueError:
                if elem != "": # 
                    res.append(elem)
        return res

    def decode(self, lst):
        res = []
        i=0
        while i<len(lst):
            elem = lst[i]
            if elem in ["SIGN+", "SIGN-"] or elem.startswith("+N") or elem.startswith("-N"):
                value = self.float_encoder.decode(lst[i:i+self.float_descriptor_length])[0]
                res.append(str(value))
                i += self.float_descriptor_length
            else:
                if elem not in ["<EOS>", "<PAD>"]:
                    res.append(elem)
                i+=1
        infix = " ".join(res)
        # nodes = infix.split(" | ")
        # nodes = [sp.parse_expr(node, local_dict=self.local_dict) for node in nodes]
        # nodes = [self.sympy_expr_to_tree(node) for node in nodes]
        # tree = NodeList(nodes)
        return infix

    def sympy_expr_to_tree(self, expr):
        prefix = self.sympy_to_prefix(expr)
        return self.equation_encoder.decode(prefix)

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]
        elif isinstance(expr, sp.Float):
            s = str(expr)
            return [s]
        elif isinstance(expr, sp.Rational):
            return ["mul", str(expr.p), "pow", str(expr.q), "-1"]
        elif expr == sp.EulerGamma:
            return ["euler_gamma"]
        elif expr == sp.E:
            return ["e"]
        elif expr == sp.pi:
            return ["pi"]

        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)

        # Unknown operator
        return self._sympy_to_prefix(str(type(expr)), expr)
    
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Mod: "mod",
        sp.Pow: "pow",
        # Misc
        sp.Abs: "abs",
        sp.sign: "sign",
        sp.Heaviside: "step",
        # Exp functions
        sp.exp: "exp",
        sp.log: "log",
        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        # Trigonometric Inverses
        sp.asin: "arcsin",
        sp.acos: "arccos",
        sp.atan: "arctan",
    }
    local_dict = {v:k for k,v in SYMPY_OPERATORS.items()}