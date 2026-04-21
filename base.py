import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from abc import ABC, abstractmethod

SAMPLE_SIZES = [10, 100, 1000, 10000]
OUTPUT_DIR = "histograms"

class BaseDistribution(ABC):
    """Abstract the base/blueprint for all distributions
    Each distribution subclass does the following functions:
    -
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generate n random samples"""""

    @abstractmethod
    def label(self) -> str:
        """Short distribution name for titles and files"""

    @abstractmethod
    def description(self) -> str:
        """Parameter description to be used in histogram subtitles"""

    def plot(self) -> str:
        """Generate histograms for all sample sizes and save as a single PNG.
        Returns the file path of the saved figure."""

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"{self.label()}  —  {self.description()}  (seed={self.seed})",
            fontsize=14, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        for idx, n in enumerate(SAMPLE_SIZES):
            self.rng = np.random.default_rng(self.seed)  # re-seed for reproducibility
            samples = self.generate(n)

            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # Discrete distributions get unit-width bins; continuous use sqrt rule
            if np.issubdtype(samples.dtype, np.integer):
                lo, hi = int(samples.min()), int(samples.max())
                bins = np.arange(lo - 0.5, hi + 1.5, 1)
            else:
                bins = min(max(int(np.sqrt(n)), 10), 60)

            ax.hist(
                samples, bins=bins, density=True,
                color="#1a73e8", edgecolor="white", linewidth=0.4, alpha=0.85,
            )
            ax.set_title(f"n = {n:,}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.text(
                0.97, 0.95,
                f"mean={samples.mean():.3f}\nstd={samples.std():.3f}",
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8),
            )

        safe_name = (self.label().lower()
                     .replace(" ", "_").replace("(", "").replace(")", ""))
        filepath = os.path.join(OUTPUT_DIR, f"hist_{safe_name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath