from sound_generator.visualize import plot_waveforms
import numpy as np
from sound_generator import soundnet, data_processing, global_configuration
from function_generator import SineGenerator, UniformSampler
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Generate Dataset
    frequencies = np.arange(10, 100, 10, dtype=float)
    uniform_sampler = UniformSampler(global_configuration.SAMPLE_FREQUENCY)
    frequency_labels = data_processing.normalize(frequencies)
    samples = [np.array(uniform_sampler.sample(SineGenerator(f))) for f in frequencies]
    padding, magnitudes, phases, params = data_processing.samples_to_training_data(
        samples
    )

    # now build and train the model
    autoencoder, encoder, decoder = soundnet.build_model(magnitudes.shape[1:], 64)
    print(autoencoder.summary())
    soundnet.train_autoencoder(autoencoder, magnitudes, frequency_labels)

    # plot how well the autoencoder does
    # TODO split into training and validation
    predictions = autoencoder.predict(magnitudes)
    recovered = []
    frequency_prediction = []
    for n, (magnitudes, f, phases, params) in enumerate(
        zip(predictions[0], predictions[1], phases, params)
    ):
        # TODO maybe move this postprocessing into a function
        recovered.append(data_processing.postprocess(magnitudes, phases, params, padding))
        frequency_prediction.append(f*90+10)
    
    plot_waveforms(samples,recovered,frequencies,frequency_prediction)