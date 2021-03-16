import numpy as np
import soundnet
import matplotlib.pyplot as plt

"""
{
    "python.formatting.provider": "black"
}
"""
SAMPLE_FREQUENCY = 2 ** 8


def sample(f, d=1, sf=SAMPLE_FREQUENCY):
    """
    Samples function f for duration d at sampling frequency sf

    Args:
        f: function that gives amplitude between -1 and 1 at time t
        d: duration of the sample
        sf: sampling frequency

    Returns: numpy array of sampled audio

    """
    t = np.linspace(0, d, int(sf * d))
    return np.vectorize(f)(t)


def sample_sin(frequency, duration=1):
    return sample(lambda t: np.sin(t * 2 * np.pi * frequency), duration)

def normalize_complex(arr):
    a = np.abs(arr)
    ma = np.max(a)
    arr /=ma
    return arr

if __name__ == "__main__":

    freq_labels = np.arange(10, 100, 10, dtype=float)
    # TODO samplefrequency
    samples = np.array([normalize_complex(np.fft.rfft(sample_sin(f)))[:-1] for f in freq_labels]).reshape(
        (-1, SAMPLE_FREQUENCY//2, 1)
    )
    freq_labels /= 100
    plt.plot(abs(samples[8, :, 0]), label = "original")
    autoencoder, encoder, decoder = soundnet.build_model((SAMPLE_FREQUENCY//2, 1), 20)
    print(autoencoder.summary())
    soundnet.train_autoencoder(autoencoder, samples, freq_labels)
    predictions = autoencoder.predict(samples)
    plt.plot(abs(predictions[0][0,..., 0]), label="predicted")
    plt.legend()
    plt.savefig("sin.png")
