import matplotlib.pyplot as plt
import numpy as np
import random
from sound_generator.global_configuration import (
    MAX_SOUNDS_PER_PAGE, SAMPLE_FREQUENCY,
    AUDIO_OUT_SAMPLE_RATE,
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

    # for more than MAX_SOUNDS_PER_PAGE samples, MAX_SOUNDS_PER_PAGE samples get randomly selected
    indices = list(range(MAX_SOUNDS_PER_PAGE))
    if size > MAX_SOUNDS_PER_PAGE:
        indices = sorted(random.sample(range(size),MAX_SOUNDS_PER_PAGE))
        size = MAX_SOUNDS_PER_PAGE
    
    # build smallest possible grid
    width = int(np.sqrt(size))
    height = int(np.ceil(size / width))
    fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(15, 15))

    # time is similar for all
    t = np.linspace(*t_range, SAMPLE_FREQUENCY)


    for n in range(size):
        os = original_sounds[indices[n]]
        ps = predicted_sounds[indices[n]]
        of = original_frequencies[indices[n]]
        pf = predicted_frequencies[indices[n]]

        x = n % width
        y = n // width
        ax = axes[y][x]
        # oops hacky rawr xd
        ax.index = (x, y)
        ax.plot(t, os, color="orange")
        ax.plot(t, ps, color="blue")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"{int(of)}Hz, predicted: {int(pf)}Hz")

    def on_click(event):
        if event.inaxes is None:
            return
        x, y = event.inaxes.index
        i = y * width + x
        if event.button == MouseButton.LEFT:
            simpleaudio.play_buffer(
                (original_sounds[indices[i]] * 2 ** 14).astype(np.int16),
                1,
                2,
                AUDIO_OUT_SAMPLE_RATE,
            )
        if event.button == MouseButton.RIGHT:
            simpleaudio.play_buffer(
                (predicted_sounds[indices[i]] * 2 ** 14).astype(np.int16),
                1,
                2,
                AUDIO_OUT_SAMPLE_RATE,
            )

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()