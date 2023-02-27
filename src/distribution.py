from typing import *
import numpy as np


class Distribution:
    def __init__(self, name: str,
                 generate: Callable[[int], np.ndarray],
                 prob_density: Callable[[float], float],
                 cumulative_func: Callable[[float], float],
                 quantile: Callable[[float], float],
                 draw_interval: Tuple[float, float] = (-4, 4),
                 discrete: bool = False) -> None:
        self.name = name
        self.generate = generate
        self.prob_density = prob_density
        self.cumulative_func = cumulative_func
        self.quantile = quantile
        self.draw_interval = draw_interval
        self.discrete = discrete
