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
        if params.float_descriptor_length == 1:
            self.float_encoder = FPSymbol(params)
        else:
            self.float_encoder = FloatSequences(params)
        self.equation_encoder = Equation(
            params, symbols, self.float_encoder, all_operators, self.constant_encoder
        )

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
        self.float_precision = params.float_precision
        self.max_exponent = params.max_exponent
        self.ndigits = (self.float_precision + 1)
        self.max_token = 10 ** self.ndigits
        self.symbols = ["+", "-", "NaN"]
        self.float_descriptor_length = params.float_descriptor_length

        if self.float_descriptor_length == 3:
            self.symbols.extend(["N" + f"%0{self.ndigits}d" % i for i in range(self.max_token)])
        elif self.float_descriptor_length == 2:
            self.symbols.extend(["N+" + f"%0{self.ndigits}d" % i for i in range(self.max_token)])
            self.symbols.extend(["N-" + f"%0{self.ndigits}d" % i for i in range(self.max_token)])
        self.symbols.extend(
            ["E" + str(i) for i in range(-self.max_exponent, self.max_exponent + 1)]
        )

    def encode(self, values):
        """
        Write a float number
        """
        precision = self.float_precision
        if isinstance(values, float) or isinstance(values, str):
            values = [values]
        values = np.array(values)
        if len(values.shape) == 1:
            seq = []
            for value in values:
                if np.isnan(value):
                    return ["NaN"]*self.float_descriptor_length
                if isinstance(value, str):
                    sign = '-' if value.startswith('-') else '+'
                    mantissa, expon = value.lstrip('-').split('e')
                    mantissa = mantissa.replace('.', '')
                    expon = int(expon)
                else:
                    sign = "+" if value >= 0 else "-"
                    m, e = (f"%.{precision}e" % value).split("e")
                    i, f = m.lstrip("-").split(".")
                    mantissa = i + f
                    expon = int(e) - precision
                    if expon < -100:
                        mantissa = "0"*self.ndigits
                        expon = int(0)
                if self.float_descriptor_length == 3:
                    token_sequence = [sign, f"N{mantissa}", f"E{expon}"]
                else:
                    token_sequence = [f"N{sign}{mantissa}",f"E{expon}"]
                seq += token_sequence
            return seq
        else:
            seqs = [self.encode(value) for value in values]
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
            for x in val:
                if x[0] not in ["-", "+", "E", "N"]:
                    return np.nan
            try:
                if self.float_descriptor_length == 2:
                    sign = 1 if val[0][1] == "+" else -1
                    mant = val[0][2:]
                    exp = int(val[1][1:])
                else:
                    sign = 1 if val[0] == "+" else -1
                    mant = val[1][1:]
                    exp = int(val[-1][1:])
                mant = int(mant)
                value = sign * float(f"{mant}e{exp}")
            except Exception:
                #import traceback
                #print(traceback.format_exc())
                #print(val)
                value = np.nan
            seq.append(value)
        return seq
    
class FPSymbol(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.float_precision = params.float_precision
        self.max_exponent = 3
        assert (self.float_precision + self.max_exponent) % 2 == 0
        self.symbols = ["NaN", "-NaN"]
        self.ndigits = (self.float_precision + 1)
        dig = 10 ** self.float_precision
        self.logrange = (self.float_precision + self.max_exponent) // 2
        self.base = 10 ** (self.logrange - self.float_precision)
        self.limit = 10 ** self.logrange
        self.output_length = 1
        # less than 1
        self.symbols.extend(["N" + str(i) + "E0" for i in range(-dig + 1, dig)])
        for i in range(self.max_exponent):
            for k in range(10**self.ndigits):
                self.symbols.append("N+" + str(k) + "E" + str(i))
                self.symbols.append("N-" + str(k) + "E" + str(i))

    def encode(self, values):

        if isinstance(values, float):
            values = [values]
        values = np.array(values)
        if len(values.shape) == 1:
            res = []
            for value in values:
                if np.isnan(value):
                    return ["NaN"]
                if abs(value) > self.limit:
                    return ["NaN"] if value > 0 else ["-NaN"]
                sign = -1 if value < 0 else 1
                v = abs(value) * self.base
                if v == 0:
                    return ["N0E0"]
                e = int(math.log10(v))
                if e < 0:
                    e = 0
                m = int(v * (10 ** (self.float_precision - e)) + 0.5)
                if m == 0:
                    sign = 1
                if m == 10 ** self.ndigits:
                    m = int(m/10)
                    e += 1
                if e >= self.max_exponent:
                    return ["NaN"] if value > 0 else ["-NaN"]
                pref = "N+" if sign == 1 else "N-"
                res.append(pref + str(m) + "E" + str(e))
            return res
        else:
            return [self.encode(v) for v in values]

    def decode(self, lst):
        res = []
        for value in lst:
            if value == "NaN":
                return self.limit, 1
            if value == "-NaN":
                return -self.limit, 1
            if value[0] != "N":
                return np.nan, 1
            m, e = value[1:].split("E")
            v = (int(m) * (10 ** int(e))) / self.limit
            res.append(v)
        return res


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
        self.float_descriptor_length = params.float_descriptor_length

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
        elif lst[0] == "+" or lst[0] == "-" or lst[0].startswith("+N") or lst[0].startswith('-N') or lst[0].startswith('N'):
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
