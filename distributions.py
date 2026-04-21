import numpy as np
from base import BaseDistribution
SAMPLE_SIZES = [10, 100, 1000, 10000]


##### DISTRIBUTIONS

class UniformInt(BaseDistribution):
    """Discrete uniform distribution over [low, high]"""

    def __init__(self, seed: int, low: int, high: int):
        super().__init__(seed)
        self.low = low
        self.high = high

    def generate(self, n: int) -> np.ndarray:
        return self.rng.integers(self.low, self.high + 1, size=n)

    def label(self) -> str:
        return "Uniform (int) Distribution"

    def description(self) -> str:
        return f"min={self.low}, max={self.high}"


class UniformReal(BaseDistribution):
    """Continuous uniform distribution over [low, high["""

    def __init__(self, seed: int, low: float, high: float):
        super().__init__(seed)
        self.low = low
        self.high = high

    def generate(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, size=n)

    def label(self) -> str:
        return "Uniform (Real) Distribution"

    def description(self) -> str:
        return f"min={self.low}, max={self.high}"

class Normal(BaseDistribution):
    """Normal or Gaussian distribution"""

    def __init__(self, seed: int, mean: float, std: float):
        super().__init__(seed)
        self.mean = mean
        self.std = std

    def generate(self, n: int) -> np.ndarray:
        return self.rng.normal(self.mean, self.std, size=n)

    def label(self) -> str:
        return "Normal Distribution"

    def description(self) -> str:
        return f"μ={self.mean}, σ={self.std}"

class Exponential(BaseDistribution):
    """Exponential distribution"""

    def __init__(self, seed: int, lambda_exp: float):
        super().__init__(seed)
        self.lambda_exp = lambda_exp

    def generate(self, n: int) -> np.ndarray:
        return self.rng.exponential(scale=self.lambda_exp, size=n)

    def label(self) -> str:
        return "Exponential Distribution"

    def description(self) -> str:
        return f" (λ={1 / self.lambda_exp:.4g})"


class Binomial(BaseDistribution):
    """Binomial distribution — Group 1 'other' distribution."""

    def __init__(self, seed: int, n_trials: int = 10, p: float = 0.5):
        super().__init__(seed)
        self.n_trials = n_trials
        self.p = p

    def generate(self, n: int) -> np.ndarray:
        return self.rng.binomial(self.n_trials, self.p, size=n)

    def label(self) -> str:
        return "Binomial"

    def description(self) -> str:
        return f"n={self.n_trials}, p={self.p}"
