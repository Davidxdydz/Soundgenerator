import numpy as np

SAMPLE_FREQUENCY = 2 ** 14
WINDOW_SIZE = SAMPLE_FREQUENCY // 2 ** 3
TRAINING_EPOCHS = 200
EMBEDDED_SIZE = 2
FEATURE_FIT_ITERATIONS = 200
available_sample_rates = np.array(
    [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]
)
AUDIO_OUT_SAMPLE_RATE = available_sample_rates[
    np.argmin(np.abs(available_sample_rates - SAMPLE_FREQUENCY))
]
print(f"Replay Audio Sample Rate: {AUDIO_OUT_SAMPLE_RATE}")