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

    # plotting for reference
    # TODO move this
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
    for n, sample in enumerate(samples):
        ax = axes[n % 3][n // 3]
        ax.plot(sample, color="orange")

    # now build and train the model
    autoencoder, encoder, decoder = soundnet.build_model(magnitudes.shape[1:], 64)
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
        recovered.append(data_processing.postprocess(magnitudes[:-padding], phases, params))
        ax = axes[n % 3][n // 3]
        ax.plot(recovered[-1], color="blue")
        ax.set_title(f"Predicted: {int(f*90+10)}Hz")
    plt.legend(["original", "predicted"])
    plt.tight_layout()
    plt.savefig("sinetest.png")
    plt.show()
    # TODO why is reconstructed always smaller in amplitude? normalization or architecture
