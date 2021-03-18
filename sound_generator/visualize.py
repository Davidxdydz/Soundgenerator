import matplotlib.pyplot as plt
import numpy as np
from sound_generator.global_configuration import SAMPLE_FREQUENCY


def plot_waveforms(
    original_sounds,
    predicted_sounds,
    original_frequencies,
    predicted_frequencies,
    t_range=(0, 1),
):
    """
    plots orignal sounds against predicted sounds in a new window

    Args:
        original_sounds: List of original sounds
        predicted_souds: List of predicted sounds
        original_frequencies: List of original frequencies
        predicted_frequencies: List of predicted frequencies

    Returns:
        Nothing
    """
    size = len(original_sounds)
    width = int(np.sqrt(size))
    height = int(np.ceil(size // width))
    _, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(15, 15))
    t = np.linspace(*t_range, SAMPLE_FREQUENCY)
    for n, (os, ps, of, pf) in enumerate(
        zip(
            original_sounds,
            predicted_sounds,
            original_frequencies,
            predicted_frequencies,
        )
    ):
        x = n % width
        y = n // width
        ax = axes[y][x]
        ax.plot(t, os, color="orange")
        ax.plot(t, ps, color="blue")
        ax.set_title(f"{int(of)}Hz, predicted: {int(pf)}Hz")
    plt.legend()
    plt.show()