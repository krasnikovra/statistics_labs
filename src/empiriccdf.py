import matplotlib.pyplot as plt
from distribution import Distribution
import numpy as np
import os


def draw_empiric_cdf(distribution: Distribution, size: int, folder: str = ".", ext: str = "pdf"):
    data = distribution.generate(size)

    def empiric_cdf(x: float):
        less_count = 0
        for z in data:
            if z < x:
                less_count += 1
        return less_count / size
    a, b = distribution.draw_interval
    x = np.linspace(a, b, 1000)

    plt.clf()
    plt.plot(x, [empiric_cdf(x) for x in x], label="Empiric CDF", color="black")
    plt.plot(x, [distribution.cumulative_func(x) for x in x], label="Theoretic CDF")
    plt.title(f"{distribution.name.capitalize()}, $n = {size}$")
    plt.xlabel(f"$x$")
    plt.ylabel("$F(x)$")
    plt.legend()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/empiriccdf_{distribution.name.casefold().replace(' ', '_')}_{size}.{ext}")
