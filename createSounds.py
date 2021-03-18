import numpy as np
from numpy.lib.scimath import log2
import soundnet
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from function_generator import SineGenerator, UniformSampler

"""
{
    "python.formatting.provider": "black"
}
"""
SAMPLE_FREQUENCY = 2 ** 10
WINDOW_SIZE = SAMPLE_FREQUENCY // 2 ** 3


def normalize(arr):
    """
    Normalizes the elements in arr to [0,1]

    Args:
        arr: the array to be normalized

    Returns:
        arrn: a normalized copy of arr
    """
    mi = np.min(arr)
    ma = np.max(arr)
    return (arr - mi) / (ma - mi)


def complex_to_polar(arr):
    """
    Converts an array of complex numbers to polar coordinats

    Args:
        arr: array of complex numbers

    Returns:
        m: the magniudes of the complex numbers
        p: the phases of the complex numbers
    """
    return np.abs(arr), np.arctan2(np.imag(arr), np.real(arr))


def polar_to_complex(magnitudes, phases):
    """
    Converts array of magnitudes and phases to an array of complex numbers

    Args:
        magnitudes: array of magnitudes
        phases: array of corrseponding phases in radians

    Returns:
        c: array of complex numbers
    """
    return magnitudes * np.exp(1j * phases)


def preprocess(sampled_sound):
    """
    Converts a sampled time series to a normalized stft representation with WINDOW_SIZE.

    Args:
        sampled_sound: array of sound sampled at SAMPLE_FREQUENCY normalized to [-1,1]

    Returns:
        normalized_magnitudes: (m,n) array of real valued magnitudes at different times (m) and frequencies (n),
                                normalized logarithmically to [-1,1]
        phases: (m,n) array of phases for reconstruction of the initial complex valued stft
        params = (min,max): parameters used for normalization, needed for denormalization
    """
    # 50% overlapping window
    _, _, zxx = stft(sampled_sound, fs=SAMPLE_FREQUENCY, nperseg=WINDOW_SIZE)
    # use only the magnitudes for the NN, add the phases back in afterwards
    # TODO find another way, this limits the flexibility
    #       maybe train on phases as well
    magnitudes, phases = complex_to_polar(zxx)
    normalized_magnitudes = np.log(magnitudes)
    # TODO maybe add treshold for very low magnitudes
    # TODO maybe normalize per timestep
    mi = np.min(normalized_magnitudes)
    ma = np.max(normalized_magnitudes)
    normalized_magnitudes = (normalized_magnitudes - mi) / abs(ma - mi) * 2 - 1
    return normalized_magnitudes, phases, (mi, ma)


def postprocess(magnitudes, phases, normalization_params,padding = None):
    """
    Converts a normalized stft representation into a denormalized time series representation.
    The stft has to represent a signal sampled at SAMPLE_FREQUENCY

    Args:
        magnitudes: (m,n) array of the magnitudes of the stft in polar coordinates
        phases: (m,n) array of the corresponding phase in radians
        normalization_params = (min,max): params to be used for denormalization

    Returns:
        x: 1D array of sample points of the stft in the time domain
    """
    mi, ma = normalization_params
    unpadded = magnitudes[:-padding] if padding else magnitudes
    denormalized_magnitudes = np.exp((unpadded + 1) / 2 * abs(ma - mi) + mi)
    zxx = polar_to_complex(denormalized_magnitudes, phases)
    _, x = istft(zxx, fs=SAMPLE_FREQUENCY)
    return x


def test_conversion():
    """
    Tests if postprocess reverts preprocess

    Returns:
        True if postprocess reverts preprocess a single example
    """

    sin_gen = SineGenerator(100)
    sampler = UniformSampler(SAMPLE_FREQUENCY)

    x1 = np.array(sampler.sample(sin_gen))
    x2 = postprocess(*preprocess(x1))
    # allclose because of floating point math
    return np.allclose(x1, x2)


def fitting_power_of_two(x):
    """
    Calculates the smallest power of two, that x is smaller or equal to

    Args:
        x: a positive value

    Returns:
        y: int, a power of two
    """
    if int(log2(x)) == log2(x):
        return x
    return 2 ** (int(log2(x)) + 1)


def samples_to_training_data(samples):
    """
    Converts a list of sampled sounds to normalized and padded training data

    Args:
        samples: Array of time series sampled data. values in [-1,1]

    Returns:
        padding: amount of zero padding added
        magnitudes: (m,n,k) array of frequency domain magnitude training data.
                m is number of sounds, n is a power of two and represents duration
        phases: (m,i,k) array of phases for sound reconstruction. i is the unpadded length
        params: array of normalization params for every sound
    """
    magnitudes = []
    # for reconstruction
    phases = []
    params = []

    # transform time domain to stft frequency domain
    for mag, pha, par in map(preprocess, samples):
        # magnitudes has shape (m,n)
        magnitudes.append(mag)
        phases.append(pha)
        params.append(par)

    po2 = fitting_power_of_two(magnitudes[0].shape[0])
    # keep to_short for going back to time domain
    padding = po2 - magnitudes[0].shape[0]
    for n, mag in enumerate(magnitudes):
        # pad the data with 0 to the next power of two so the network has the right dimensions
        magnitudes[n] = np.pad(mag, [(0, padding), (0, 0)])
    magnitudes = np.array(magnitudes)

    return padding, magnitudes, phases, params


# TODO move most of this to another file, soundgeneration should not need to import all of tensorflow
if __name__ == "__main__":

    # Tests
    assert (
        test_conversion()
    ), "sound -> preprocessing -> postprocessing does not equal original sound"

    # Generate Dataset
    frequencies = np.arange(10, 100, 10, dtype=float)
    uniform_sampler = UniformSampler(SAMPLE_FREQUENCY)
    frequency_labels = normalize(frequencies)
    samples = [np.array(uniform_sampler.sample(SineGenerator(f))) for f in frequencies]
    padding, magnitudes, phases, params = samples_to_training_data(samples)

    # plotting for reference
    # TODO move this
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
    for n, sample in enumerate(samples):
        ax = axes[n % 3][n // 3]
        ax.plot(sample, color="orange")

    # now build and train the model
    autoencoder, encoder, decoder = soundnet.build_model(magnitudes.shape[1:], 20)
    print(autoencoder.summary())
    soundnet.train_autoencoder(autoencoder, magnitudes, frequency_labels)

    # plot how well the autoencoder does
    # TODO split into training and validation
    predictions = autoencoder.predict(magnitudes)
    recovered = []
    for n, (magnitudes, f, phases, params) in enumerate(
        zip(predictions[0], predictions[1], phases, params)
    ):
        # TODO maybe move this postprocessing into a function
        recovered.append(postprocess(magnitudes, phases, params,padding))
        ax = axes[n % 3][n // 3]
        ax.plot(recovered[-1], color="blue")
        ax.set_title(f"Predicted: {int(f*90+10)}Hz")
    plt.legend(["original", "predicted"])
    plt.tight_layout()
    plt.savefig("sinetest.png")
    plt.show()
    # TODO why is reconstructed always smaller in amplitude? normalization or architecture
