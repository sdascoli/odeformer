# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, List, Union
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
from sympy import Add, preorder_traversal
import regex
import numpy as np
import scipy
import sympy
from odeformer.envs.generators import Node, NodeList

def get_complexity(expr: sympy.core.Expr):
    # taken from: https://github.com/cavalab/srbench/blob/master/postprocessing/symbolic_utils.py#L12:L16
    c=0
    for arg in preorder_traversal(expr):
        c += 1
    return c

def tokenize_equation(eq: str, tokens_to_ignore: str = "|,", debug: bool = False) -> List[str]:
    constant_token = "€"
    tokens_to_ignore = tokens_to_ignore.split(",")
    assert constant_token not in tokens_to_ignore
    tokens = []
    eq = eq.replace(" ", "")
    CONSTANTS_PATTERN=r"(?:(?<!_\d*))(?:(?<=[\*/\+-])[-+]?)?(?:(?<=\()[-+]?)?(?:(?<=^)[-+]?)?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
    constants = regex.findall(pattern=CONSTANTS_PATTERN, string=eq)
    eq = regex.sub(pattern=CONSTANTS_PATTERN, repl=str(constant_token), string=eq)
    eq = eq.replace(" ","")
    constant_i = 0
    op = ""
    for symbol in eq:
        if symbol in tokens_to_ignore:
            if debug: print(symbol, "ignore", op)
            continue
        elif symbol in ["+", "-", "*", "/"]:
            if debug: print(symbol, '["+", "-", "*", "/"]', op)
            if op != "":
                tokens.append(op)
                op = ""
            tokens.append(symbol)
        elif symbol == constant_token:
            if debug: print(symbol, 'constant_token', op)
            if op != "":
                tokens.append(op)
                op = ""
            tokens.append(constants[constant_i])
            constant_i += 1
        elif symbol == "(":
            if debug: print(symbol, '(', op)
            if op != "":
                tokens.append(op)
                op = ""
        elif symbol == ")":
            if debug: print(symbol, ')', op)
            if op != "":
                tokens.append(op)
                op = ""
        else:
            if debug: print(symbol, 'else', op)
            op += symbol
            
        if debug: print(tokens, op)
    if op != "":
        tokens.append(op)
        op = ""
    return tokens

def compute_metrics(predicted, true, predicted_tree=None, tree=None, metrics="r2"):
    results = defaultdict(list)
    if metrics == "":
        return {}
    if len(true.shape)<3: # we are dealing with a single trajectory
        predicted, true, predicted_tree, tree = [predicted], [true], [predicted_tree], [tree]
    assert len(true) == len(predicted), "issue with len, true: {}, predicted: {}".format(len(true), len(predicted))

    for metric in metrics.split(","):

        if metric == "is_valid_tree":
            for i in range(len(true)):
                if predicted_tree[i] is None:
                    results[metric].append(0)
                else: 
                    results[metric].append(1)

        elif metric == "is_valid":
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(0)
                else: 
                    results[metric].append(1)

        elif metric.startswith("r2"):
            if metric == "r2":
                for i in range(len(true)):
                    if predicted[i] is None or np.isnan(np.min(predicted[i])):
                        results[metric].append(0)
                    else:
                        try:
                            results[metric].append(r2_score(true[i], predicted[i], multioutput='variance_weighted'))
                        except Exception as e:
                            results[metric].append(0)
            elif metric == "r2_zero":
                for i in range(len(true)):
                    if predicted[i] is None or np.isnan(np.min(predicted[i])):
                        results[metric].append(0)
                    else:
                        try:
                            results[metric].append(max(0, r2_score(true[i], predicted[i], multioutput='variance_weighted')))
                        except Exception as e:
                            results[metric].append(0)
            elif metric.startswith("r2_zero_dim"):
                dimension = int(metric.split("_")[-1])
                for i in range(len(true)):
                    if predicted[i] is None or np.isnan(np.min(predicted[i])):
                        results[metric].append(0)
                    else:
                        try:
                            results[metric].append(max(0, r2_score(true[i], predicted[i], multioutput='raw_values')[dimension]))
                        except Exception as e:
                            results[metric].append(0)

        elif metric.startswith("accuracy_l1"):
            if metric == "accuracy_l1":
                atol, rtol = 0.0, 0.1
                tolerance_point = 0.95
            elif metric == "accuracy_l1_biggio":
                ## default is biggio et al.
                atol, rtol = 1e-3, 0.05
                tolerance_point = 0.95
            else:
                atol = 0 #float(metric.split("_")[-3])
                rtol = float(metric.split("_")[-1])
                tolerance_point = 0.95 #float(metric.split("_")[-1])

            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(0)
                else:
                    try:
                        # print("predicted[i], true[i]", predicted[i], true[i])
                        is_close = np.isclose(predicted[i], true[i], atol=atol, rtol=rtol)
                        results[metric].append(float(is_close.mean()>=tolerance_point))
                    except Exception as e:
                        results[metric].append(0)

        elif metric == "snmse":
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        dimension = predicted[i].shape[1]
                        snmse = sum([np.sqrt(mean_squared_error(true[i][:,dim], predicted[i][:,dim])/(np.var(true[i][:,dim])+1.e-10)) for dim in range(dimension)])
                    except: 
                        snmse = np.nan
                    results[metric].append(snmse)
                        
        elif metric == "mse":
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        results[metric].append(mean_squared_error(true[i], predicted[i]))
                    except Exception as e:
                        results[metric].append(np.nan)

        elif metric == "nmse":
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        mean_y = np.mean(true[i])
                        NMSE = (np.mean(np.square(true[i]- predicted[i])))/mean_y
                        results[metric].append(NMSE)
                    except Exception as e:
                        results[metric].append(np.nan)

        elif metric == "rmse":
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        results[metric].append(mean_squared_error(true[i], predicted[i], squared=False))
                    except Exception as e:
                        results[metric].append(np.nan)

        elif metric == "is_symbolic_solution":
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        diff = true[i] - predicted[i]
                        div = true[i] / (predicted[i] + 1e-100)
                        std_diff = scipy.linalg.norm(
                            np.abs(diff - diff.mean(0))
                        )
                        std_div = scipy.linalg.norm(
                            np.abs(div - div.mean(0))
                        )
                        if std_diff<1e-10 and std_div<1e-10: results[metric].append(1.0)
                        else: results[metric].append(0.0)
                    except Exception as e:
                        results[metric].append(np.nan)

        elif metric == "l1_error":
            
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        l1_error = np.mean(np.abs((true[i] - predicted[i])))
                        if np.isnan(l1_error): results[metric].append(np.infty)
                        else: results[metric].append(l1_error)
                    except Exception as e:
                        results[metric].append(np.nan)

        elif metric == "complexity":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(len(predicted_tree[i].prefix().replace("|", "").split(",")))
                    
        elif metric == "relative_complexity":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(
                        len(predicted_tree[i].prefix().replace("|", "").split(",")) - \
                            len(tree[i].prefix().replace("|", "").split(","))
                    )

        elif metric == "complexity_sympy":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for ptree in predicted_tree:
                if ptree is None:
                    results[metric].append(np.nan)
                else:
                    if not isinstance(ptree, str):
                        # TODO: are predicted_tree from Odeformer already in infix format? How to get them as string?
                        ptree = ptree.infix()
                    results[metric].append(
                        np.sum([get_complexity(sympy.parse_expr(comp)) for comp in ptree.split("|")])
                    )
        
        elif metric == "relative_complexity_sympy":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for ptree, gttree in zip(predicted_tree, tree):
                if ptree is None or gttree is None:
                    if gttree is None:
                        print("Cannot compute relative_complexity_sympy as ground truth tree is None. Returning np.nan")
                    results[metric].append(np.nan)
                else:
                    if not isinstance(ptree, str):
                        ptree = ptree.infix()
                    if not isinstance(gttree, str):
                        gttree = gttree.infix()
                    results[metric].append(
                        np.sum([get_complexity(sympy.parse_expr(comp)) for comp in ptree.split("|")]) - \
                            np.sum([get_complexity(sympy.parse_expr(comp)) for comp in gttree.split("|")])
                    )

        elif metric == "complexity_string":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for ptree in predicted_tree:
                if ptree is None:
                    results[metric].append(np.nan)
                else:
                    if not isinstance(ptree, str):
                        # TODO: are predicted_tree from Odeformer already in infix format? How to get them as string?
                        ptree = ptree.infix()
                    results[metric].append(
                        np.sum([
                            len(tokenize_equation(str(sympy.parse_expr(comp))))
                            for comp in ptree.split("|")
                        ])
                    )
                    
        elif metric == "relative_complexity_string":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for ptree, gttree in zip(predicted_tree, tree):
                if ptree is None or gttree is None:
                    if gttree is None:
                        print("Cannot compute relative_complexity_sympy as ground truth tree is None. Returning np.nan")
                    results[metric].append(np.nan)
                else:
                    if not isinstance(ptree, str):
                        ptree = ptree.infix()
                    if not isinstance(gttree, str):
                        gttree = gttree.infix()
                    results[metric].append(
                        np.sum([
                            len(tokenize_equation(str(sympy.parse_expr(comp)))) 
                            for comp in ptree.split("|")
                        ]) - \
                        np.sum([
                            len(tokenize_equation(str(sympy.parse_expr(comp))))
                            for comp in gttree.split("|")
                        ])
                    )

        elif metric == "edit_distance":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None or tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    distance = min_edit_distance(predicted_tree[i].prefix(skeleton=True).split(","), tree[i].prefix(skeleton=True).split(","))
                    results[metric].append(distance)

        elif metric == "term_difference":
            if not predicted_tree: 
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None or tree[i] is None or len(predicted_tree[i].nodes) != len(tree[i].nodes):
                    results[metric].append(np.nan)
                else:
                    dimension = len(predicted_tree[i].nodes)
                    extra, missing = [], []
                    for dim in range(dimension):
                        pred_terms = predicted_tree[i].nodes[dim].infix(skeleton=True).split(" + ")
                        terms = tree[i].nodes[dim].infix(skeleton=True).split(" + ")
                        missing.append(sum([1 for term in terms if term not in pred_terms])/len(terms))
                        extra.append(sum([1 for term in pred_terms if term not in terms])/len(terms))
                    results[metric].append(np.mean(missing) + np.mean(extra))
                    
        elif metric == "term_difference_sympy":
            
            # If variables names contain indices, the index must be preceeded by "_", e.g. x_0, x_1
            
            def get_terms(eq: str, constant_token: Union[None, Literal["CONSTANT"]]=Literal["CONSTANT"]) -> List:
                terms = [str(term) for term in list(Add.make_args(sympy.parse_expr(eq)))]
                if constant_token is not None:
                    terms = [
                        regex.sub(
                            pattern=r"(?<!_\d*)[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?",
                            repl=str(constant_token),
                            string=term,
                        ) for term in terms
                    ]
                return terms
                
            if not predicted_tree:
                results[metric].extend([np.nan for _ in range(len(true))])
                continue
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None or tree[i] is None or len(predicted_tree[i].nodes) != len(tree[i].nodes):
                    results[metric].append(np.nan)
                else:
                    dimension = len(predicted_tree[i].nodes)
                    extra, missing = [], []
                    for dim in range(dimension):
                        pred_terms = get_terms(
                            eq=predicted_tree[i].nodes[dim].infix(skeleton=False), 
                            constant_token="CONSTANT",
                        )
                        terms = get_terms(
                            eq=tree[i].nodes[dim].infix(skeleton=False),
                            constant_token="CONSTANT",
                        )
                        missing.append(sum([1 for term in terms if term not in pred_terms])/len(terms))
                        extra.append(sum([1 for term in pred_terms if term not in terms])/len(terms))
                    results[metric].append(np.mean(missing) + np.mean(extra))
        else:
            raise NotImplementedError("Metric {} not implemented".format(metric))

    return results

def min_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])

    return dp[m][n]
