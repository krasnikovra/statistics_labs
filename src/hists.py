import matplotlib.pyplot as plt
import numpy as np
import os
from distribution import Distribution


def draw_hist(distribution: Distribution, size: int, folder: str = ".", ext: str = "pdf"):
    data = distribution.generate(size)

    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1 / 3)
    bins = round((data.max() - data.min()) / bin_width)

    x = np.linspace(data.min(), data.max(), 1000) if not distribution.discrete else np.arange(data.min(), data.max())

    plt.clf()
    plt.hist(data, density=True, bins=bins, label="Generated data", edgecolor='black', linewidth=1.0)
    plt.plot(x, [distribution.prob_density(x) for x in x], label="Density")
    plt.title(f"{distribution.name.capitalize()}, $n = {size}$")
    plt.xlabel(f"{distribution.name.capitalize()} numbers")
    plt.ylabel("Density")
    plt.legend()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/hist_{distribution.name.casefold().replace(' ', '_')}_{size}.{ext}")
