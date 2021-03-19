import matplotlib.pyplot as plt
import numpy as np
from sound_generator.global_configuration import (
    SAMPLE_FREQUENCY,
    NEAREST_SAMPLE_RATE_THAT_IS_NOT_WEIRD_FUCK_YOU_SIMPLE_AUDIO_WHY_CANT_I_USE_WHATEVER_THE_FUCK_I_WANT,
)
from matplotlib.backend_bases import MouseButton
import simpleaudio


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
    height = int(np.ceil(size / width))
    fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(15, 15))
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
        # oops hacky rawr xd
        ax.index = (x, y)
        ax.plot(t, os, color="orange")
        ax.plot(t, ps, color="blue")
        ax.set_title(f"{int(of)}Hz, predicted: {int(pf)}Hz")

    def on_click(event):
        if event.inaxes is None:
            return
        x, y = event.inaxes.index
        i = y * width + x
        if event.button == MouseButton.LEFT:
            simpleaudio.play_buffer(
                (original_sounds[i] * 2 ** 14).astype(np.int16),
                1,
                2,
                NEAREST_SAMPLE_RATE_THAT_IS_NOT_WEIRD_FUCK_YOU_SIMPLE_AUDIO_WHY_CANT_I_USE_WHATEVER_THE_FUCK_I_WANT,
            )
        if event.button == MouseButton.RIGHT:
            simpleaudio.play_buffer(
                (predicted_sounds[i] * 2 ** 14).astype(np.int16),
                1,
                2,
                NEAREST_SAMPLE_RATE_THAT_IS_NOT_WEIRD_FUCK_YOU_SIMPLE_AUDIO_WHY_CANT_I_USE_WHATEVER_THE_FUCK_I_WANT,
            )

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()