import numpy as np
SAMPLE_FREQUENCY = 2 ** 14
WINDOW_SIZE = SAMPLE_FREQUENCY // 2 ** 4
TRAINING_EPOCHS = 100
FEATURE_FIT_ITERATIONS = 100
available_sample_rates = np.array([8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000])
NEAREST_SAMPLE_RATE_THAT_IS_NOT_WEIRD_FUCK_YOU_SIMPLE_AUDIO_WHY_CANT_I_USE_WHATEVER_THE_FUCK_I_WANT = available_sample_rates[np.argmin(np.abs(available_sample_rates-SAMPLE_FREQUENCY))]
