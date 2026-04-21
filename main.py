from distributions import (
    UniformInt,
    UniformReal,
    Normal,
    Exponential,
    Binomial,
    SAMPLE_SIZES,
)
import numpy as np

SEED = 1


def print_samples(dist) -> None:
    """Print a summary table of generated values for each sample size."""
    print(f"\n{'=' * 60}")
    print(f"  {dist.label()}  |  {dist.description()}")
    print(f"{'=' * 60}")
    print(f"  {'n':>8}  |  {'mean':>10}  {'std':>10}  |  first 5 values")
    print(f"  {'-' * 8}  |  {'-' * 10}  {'-' * 10}  |  {'-' * 25}")

    for n in SAMPLE_SIZES:
        dist.rng = np.random.default_rng(SEED)
        samples = dist.generate(n)
        preview = "  ".join(f"{v}" for v in samples[:5])
        print(f"  {n:>8,}  |  {samples.mean():>10.4f}  {samples.std():>10.4f}  |  {preview}")


def main():
    distributions = [
        UniformInt(SEED, low=0, high=20),
        UniformReal(SEED, low=0.0, high=1.0),
        Normal(SEED, mean=0, std=1),
        Exponential(SEED, lambda_exp=1),
        Binomial(SEED, n_trials=10, p=0.5),
    ]

    print("\nPerformance Evaluation of Computer Systems — USEEJ7")
    print("Lab 1: Random Variable Generation  |  Group 1  |  Seed: ", SEED)

    for dist in distributions:
        print_samples(dist)
        path = dist.plot()
        print(f"\n  Histogram saved -> {path}")

    print(f"\n{'=' * 60}")
    print("Done")


if __name__ == "__main__":
    main()