import math

import numpy as np
from typing import *
from distribution import Distribution
import os


def mean(sorted_data: np.ndarray):
    return np.mean(sorted_data)


def var(sorted_data: np.ndarray):
    squares = np.array([x ** 2 for x in sorted_data])
    return mean(squares) - mean(sorted_data) ** 2


def med(sorted_data: np.ndarray):
    return np.median(sorted_data)


def extremal_mean(sorted_data: np.ndarray):
    return (sorted_data[0] + sorted_data[-1]) / 2


def quartiles_mean(sorted_data: np.ndarray):
    return (np.quantile(sorted_data, 0.25) + np.quantile(sorted_data, 0.75)) / 2


def truncated_mean(sorted_data: np.ndarray):
    n = len(sorted_data)
    r = int(np.round(n / 4))
    res = 0
    for i in range(r + 1, n - r + 1):
        res += sorted_data[i] / (n - 2 * r)
    return res


def generate_characteristic(distribution: Distribution,
                            characteristic: Callable[[np.ndarray], float],
                            size: int,
                            repeats: int = 1000):
    res = np.array([])
    for _ in range(repeats):
        data = distribution.generate(size)
        sorted_data = np.sort(data)
        res = np.append(res, characteristic(sorted_data))
    return res


def mean_and_var_of_characteristics(distribution: Distribution,
                                    sizes: List[int],
                                    folder: str = ".",
                                    repeats: int = 1000,
                                    ext: str = "tex"):
    chars = [mean, med, extremal_mean, quartiles_mean, truncated_mean]
    chars_heads = ["$\\overline{x}$", "$med\\ x$", "$z_R$", "$z_Q$", "$z_{tr}$"]

    upper_chars = [mean, var]
    upper_chars_heads = ["$E(z)$", "$D(z)$"]

    filename = f"{folder}/chars_{distribution.name.casefold().replace(' ', '_')}.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| ")
        for _ in range(len(chars) + 1):
            file.write("c |")
        file.write("} \\hline \n")

        for size in sizes:
            file.write(f"{distribution.name.capitalize()}, n = {size}")
            for _ in chars:
                file.write(" &")
            file.write("\\\\ \\hline \n")

            file.write(f" & {' & '.join(chars_heads)} \\\\ \\hline \n")

            mean_row = [mean(generate_characteristic(distribution, char, size, repeats))
                        for char in chars]
            file.write(f"$E(z)$ & {' & '.join([np.round(x, 4).astype(str) for x in mean_row])} \\\\ \\hline \n")
            variance_row = [var(generate_characteristic(distribution, char, size, repeats))
                            for char in chars]
            file.write(f"$D(z)$ & {' & '.join([np.round(x, 4).astype(str) for x in variance_row])} \\\\ \\hline \n")
            mean_pm_std_row = [(e - math.sqrt(d), e + math.sqrt(d)) for (e, d) in zip(mean_row, variance_row)]
            mean_pm_std_row_str = [f"\\scriptsize{{[{np.round(a, 4).astype(str)}, {np.round(b, 4).astype(str)}]}}" for (a, b) in
                                   mean_pm_std_row]
            file.write(f"$E\pm \sqrt" + "{D(z)}$" + f" & {' & '.join(mean_pm_std_row_str)} \\\\ \\hline \n")

            for _ in chars:
                file.write("& ")
            file.write("\\\\ \\hline \n")

        file.write("\\end{tabular}")
