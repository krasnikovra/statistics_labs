from distribution import Distribution
from typing import *

import matplotlib.pyplot as plt
import os


def draw_boxplot(distribution: Distribution, sizes: List[int], folder: str = ".", ext: str = "pdf"):
    plt.clf()

    data = [distribution.generate(size) for size in sizes]
    plt.boxplot(data, vert=False, widths=0.7)
    plt.yticks(ticks=[x + 1 for x in range(len(sizes))], labels=sizes)
    plt.title(f"{distribution.name.capitalize()}")
    plt.xlabel(f"{distribution.name.capitalize()} numbers")
    plt.ylabel(f"Dataset size")

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/boxplot_{distribution.name.casefold().replace(' ', '_')}.{ext}")
