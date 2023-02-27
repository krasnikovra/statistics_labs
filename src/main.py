import numpy as np

import scipy.stats as sci

from boxplots import draw_boxplot
from characteristics import mean_and_var_of_characteristics
from distribution import Distribution
from empiriccdf import draw_empiric_cdf
from fliers import write_fliers
from hists import draw_hist
from kernels import draw_kernel_approx
from theoretic_fliers import theoretic_fliers

distributions = [
    Distribution(
        name="normal",
        generate=lambda size: sci.norm.rvs(loc=0, scale=1, size=size),
        prob_density=lambda x: sci.norm.pdf(x, loc=0, scale=1),
        cumulative_func=lambda x: sci.norm.cdf(x, loc=0, scale=1),
        quantile=lambda x: sci.norm.ppf(x, loc=0, scale=1),
    ),
    Distribution(
        name="cauchy",
        generate=lambda size: sci.cauchy.rvs(loc=0, scale=1, size=size),
        prob_density=lambda x: sci.cauchy.pdf(x, loc=0, scale=1),
        cumulative_func=lambda x: sci.cauchy.cdf(x, loc=0, scale=1),
        quantile=lambda x: sci.cauchy.ppf(x, loc=0, scale=1)
    ),
    Distribution(
        name="laplace",
        generate=lambda size: sci.laplace.rvs(loc=0, scale=1 / np.sqrt(2), size=size),
        prob_density=lambda x: sci.laplace.pdf(x, loc=0, scale=1 / np.sqrt(2)),
        cumulative_func=lambda x: sci.laplace.cdf(x, loc=0, scale=1 / np.sqrt(2)),
        quantile=lambda x: sci.laplace.ppf(x, loc=0, scale=1 / np.sqrt(2))
    ),
    Distribution(
        name="poisson",
        generate=lambda size: sci.poisson.rvs(10, loc=0, size=size),
        prob_density=lambda x: sci.poisson.pmf(x, 10, loc=0),
        cumulative_func=lambda x: sci.poisson.cdf(x, 10, loc=0),
        quantile=lambda x: sci.poisson.ppf(x, 10, loc=0),
        draw_interval=(6, 14),
        discrete=True
    ),
    Distribution(
        name="uniform",
        generate=lambda size: sci.uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=size),
        prob_density=lambda x: sci.uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        cumulative_func=lambda x: sci.uniform.cdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        quantile=lambda x: sci.uniform.ppf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
    )
]


def hists():
    for size in [10, 50, 1000]:
        for distribution in distributions:
            draw_hist(distribution, size, folder="../figures/hists")


def chars():
    for distribution in distributions:
        mean_and_var_of_characteristics(distribution, [10, 100, 1000], folder="../figures/chars")


def boxplots():
    for distribution in distributions:
        draw_boxplot(distribution, [20, 100], folder="../figures/boxplots")


def fliers():
    write_fliers(distributions, [20, 100], folder="../figures/fliers")


def theor_fliers():
    theoretic_fliers(distributions, folder="../figures/fliers")


def empiriccdfs():
    for distribution in distributions:
        for size in [20, 60, 100]:
            draw_empiric_cdf(distribution, size, folder="../figures/empiriccdfs")


def kernels():
    for distribution in distributions:
        for size in [20, 60, 100]:
            for h_mul in [0.5, 1.0, 2.0]:
                draw_kernel_approx(distribution, size, h_mul, folder="../figures/kernels")


if __name__ == "__main__":
    np.random.seed(73593815)

    hists()
    chars()
    boxplots()
    fliers()
    theor_fliers()
    empiriccdfs()
    kernels()
