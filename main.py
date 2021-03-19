from sound_generator.visualize import plot_waveforms
import numpy as np
from sound_generator import soundnet, data_processing, global_configuration
from function_generator import SineGenerator, UniformSampler,SawGenerator
import matplotlib.pyplot as plt


def pitch_shift(encoder, decoder, frequency_classifier, sample, target_frequency):
    """
    Retargets a sample to match a frequency target.

    Args:
        encoder: The encoder to use to obtain the embedding.
        decoder: The decoder for the embedding space.
        frequency_classifier: The classifier used to move in feature space.
        sample: The sample to retarget.
        target_frequency: The retarget frequency.

    Returns:
        The retargeted result sample.
    """

    t = soundnet.match_features_space_frequency(
        encoder, frequency_classifier, sample, target_frequency
    )
    return decoder.predict(t).shape


if __name__ == "__main__":

    # Generate Dataset
    frequencies = list(range(10, 100, 20))
    uniform_sampler = UniformSampler(global_configuration.SAMPLE_FREQUENCY)
    samples = [np.array(uniform_sampler.sample(SineGenerator(f))) for f in frequencies]
    samples += [np.array(uniform_sampler.sample(SawGenerator(f))) for f in frequencies]
    frequencies*=2

    # a real sound
    # trndsttr = np.loadtxt("samples.csv",delimiter=",")[:global_configuration.SAMPLE_FREQUENCY*2:2,0]
    # samples.append(trndsttr)
    # frequencies.append(400)

    (padding, magnitudes, phases, m_params), (frequency_labels,f_params) = data_processing.batch_preprocess(
        samples,
        frequencies
    )

    # now build and train the model
    autoencoder, encoder, decoder, frequency_classifier = soundnet.build_model(
        magnitudes.shape[1:], 64
    )
    soundnet.train_autoencoder(autoencoder, magnitudes, frequency_labels)

    # plot how well the autoencoder does
    # TODO split into training and validation
    predictions = autoencoder.predict(magnitudes)
    recovered = []
    frequency_prediction = []
    for n, (mag, fre, pha, par) in enumerate(
        zip(predictions[0], predictions[1], phases, m_params)
    ):
        # TODO maybe move this postprocessing into a function
        recovered.append(data_processing.postprocess(mag, pha, par, padding))
        frequency_prediction.append(data_processing._denormalize(fre,f_params))

    plot_waveforms(samples, recovered, frequencies, frequency_prediction)
    print(autoencoder.summary())
    # TODO Proper testing.
    pitch_shift(encoder, decoder, frequency_classifier, magnitudes[0], 0.5)