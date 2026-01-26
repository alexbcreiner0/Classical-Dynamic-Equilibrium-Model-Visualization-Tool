from dataclasses import dataclass, field
from numpy import array, ndarray


@dataclass
class Params:
    random_everything: bool = False
    alpha: ndarray = field(default_factory=lambda: array([1.0, 1.0, 1.0, 1.0]))
    dt: float = 0.001
    T: int = 1000
    A: ndarray = field(default_factory=lambda: array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
    interval_alpha: ndarray = field(default_factory=lambda: array([0.1, 0.1, 0.1, 0.1]))
    interval_mu: ndarray = field(default_factory=lambda: array([0.1, 0.1, 0.1, 0.1]))
