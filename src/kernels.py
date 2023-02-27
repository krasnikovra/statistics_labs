from distribution import Distribution
import numpy as np
import matplotlib.pyplot as plt
import os


def gauss_kernel(u: float) -> float:
    return np.exp(-(u ** 2) / 2) / np.sqrt(2 * np.pi)


def silverman_h(data: np.ndarray):
    return 1.06 * np.std(data) * (len(data) ** (-0.2))


def draw_kernel_approx(distribution: Distribution,
                       size: int,
                       h_mul: float = 1.0,
                       folder: str = ".",
                       ext: str = "pdf"):
    data = distribution.generate(size)
    K = gauss_kernel
    h = silverman_h(data) * h_mul
    n = len(data)

    def approx_density(x: float):
        res = 0
        for xi in data:
            res += K((x - xi) / h) / (n * h)
        return res

    a, b = distribution.draw_interval
    x = np.linspace(a, b, 1000) if not distribution.discrete else np.arange(a, b + 1)

    plt.clf()
    plt.plot(x, [approx_density(x) for x in x], label="Kernel approximation", color="black")
    plt.plot(x, [distribution.prob_density(x) for x in x], label="Theoretic density")
    plt.title(f"{distribution.name.capitalize()}, $n = {size}$, $h = {h_mul} \cdot h_n$")
    plt.xlabel(f"$x$")
    plt.ylabel("$f(x)$")
    plt.legend()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/kernel_{distribution.name.casefold().replace(' ', '_')}_{size}_{h_mul}h.{ext}")
