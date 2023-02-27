from distribution import Distribution

import numpy as np
from typing import *
import os


def fliers_percentage(distribution: Distribution, size: int, repeats: int = 1000) -> float:
    res = []
    for _ in range(repeats):
        data = distribution.generate(size)
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        u = q3 + 1.5 * (q3 - q1)
        l = q1 - 1.5 * (q3 - q1)
        fliers_count = 0
        for x in data:
            if x < l or x > u:
                fliers_count += 1
        res.append(fliers_count / size)
    return np.mean(res)


def write_fliers(distributions: List[Distribution],
                 sizes: List[int],
                 repeats: int = 1000,
                 folder: str = ".",
                 ext: str = "tex"):
    filename = f"{folder}/fliers.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| c | c |} \\hline\n")
        file.write("Выборка & Доля выбросов \\\\ \\hline \n")
        for distribution in distributions:
            for size in sizes:
                file.write(f"{distribution.name.capitalize()}, n = {size} & "
                           f"{np.round(fliers_percentage(distribution, size, repeats), 3)} \\\\ \\hline\n")
        file.write("\\end{tabular}")
