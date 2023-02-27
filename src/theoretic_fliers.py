from distribution import Distribution
from typing import *
import os
import numpy as np


def theoretic_flier(distribution: Distribution):
    q1 = distribution.quantile(0.25)
    q3 = distribution.quantile(0.75)
    x1 = q1 - 1.5 * (q3 - q1)
    x2 = q3 + 1.5 * (q3 - q1)
    p = distribution.cumulative_func(x1) + 1 - distribution.cumulative_func(x2) if not distribution.discrete \
        else distribution.cumulative_func(x1) - distribution.prob_density(x1) + 1 - distribution.cumulative_func(x2)
    return (q1, q3, x1, x2, p)


def theoretic_fliers(distributions: List[Distribution], folder: str = ".", ext: str = "tex"):
    filename = f"{folder}/theoretic_fliers.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| c | c | c | c | c | c |} \\hline\n")
        file.write("Распределение & $Q_1^T$ & $Q_3^T$ & $X_1^T$ & $X_2^T$ & $P_\\text{в}^T$ \\\\\\hline\n")
        for distribution in distributions:
            (q1, q3, x1, x2, p) = np.round(theoretic_flier(distribution), 3)
            file.write(f"{distribution.name.capitalize()} & {q1} & {q3} & {x1} & {x2} & {p} \\\\\\hline\n")
        file.write("\\end{tabular}")
