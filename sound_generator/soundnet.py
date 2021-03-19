from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    Flatten,
    UpSampling1D,
)
from tensorflow.keras.models import Model
from sound_generator.global_configuration import TRAINING_EPOCHS, FEATURE_FIT_ITERATIONS,EMBEDDED_SIZE
import numpy as np


def build_encoder(input_shape):
    """
    Builds an encoder model for use in an autoencoder.

    Args:
        input_shape: Tuple with network input shape

    Returns:
        Encoder model with specified dimensions.
    """

    # TODO increase params, atm only 26k params
    encoder_input = Input(input_shape)
    encoder_hidden = Conv1D(64, 4, padding="same", activation="relu")(encoder_input)
    encoder_hidden = Dense(32, activation="relu")(encoder_hidden)
    encoder_hidden = Conv1D(16, 4, padding="same", activation="relu")(encoder_hidden)

    # TODO make this constant maybe so that input shape is not correlated with embedded_size
    # (n / 16, embedded_size)
    encoder_out = Dense(EMBEDDED_SIZE, activation="relu")(encoder_hidden)

    return Model(encoder_input, encoder_out, name="Encoder")


def build_decoder(feature_shape):
    """
    Builds an decoder model for use in an autoencoder.

    Args:
        feature_shape: Tuple containing the shape of the embedded input space.

    Returns:
        Decoder model with specified dimensions.
    """

    decoder_input = Input(feature_shape)
    decoder_hidden = Conv1D(16, 4, padding="same", activation="relu")(decoder_input)
    decoder_hidden = Dense(32, activation="relu")(decoder_input)
    decoder_hidden = Conv1D(64, 4, padding="same", activation="relu")(decoder_hidden)
    decoder_out = Conv1D(1, 4, padding="same", activation="relu")(decoder_hidden)

    return Model(decoder_input, decoder_out, name="Decoder")


def build_freq_classifier(feature_shape):
    """
    Builds a classifier model to investigate frequencies in embedded space.

    Args:
        feature_shape: Tuple containing the shape of the embedded input space.

    Returns:
        A single output classifier that estimates the frequency a specific embedded
        point represents.
    """

    freq_input = Input(feature_shape)
    freq_flatten = Flatten()(freq_input)
    freq_classifier = Dense(16, activation="relu")(freq_flatten)
    freq_classifier = Dense(1, activation="relu")(freq_classifier)
    return Model(freq_input, freq_classifier, name="FreqClassifier")


def build_model(input_shape):
    """
    Builds a frequency autoencoder model with a side output for
    that estimates the frequncy of the input sample.

    Args:
        input_shape: Tuple with network input and output shape

    Returns:
        A Quadruple containing the complete autoencoder, the encoder and
        decoder and the freuqency classifier.
    """

    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.output_shape[1:])
    freq_classifier = build_freq_classifier(encoder.output_shape[1:])

    autoencoder_input = Input(input_shape)
    autoencoder_enc = encoder(autoencoder_input)
    autoencoder_freq = freq_classifier(autoencoder_enc)
    autoencoder_dec = decoder(autoencoder_enc)

    return (
        Model(
            autoencoder_input, [autoencoder_dec, autoencoder_freq], name="autoencoder"
        ),
        encoder,
        decoder,
        freq_classifier,
    )


def freeze_model(model, freeze):
    """
    Set model training state frozen or unfrozen.

    Args:
        model: The model to set the state of.
        freeze: Freeze state.
    """

    for layer in model.layers:
        layer.trainable = not freeze


def match_features_space_frequency(encoder, frequency_classifier, sample, frequency):
    """
    Retargets a sample in feature space to match the frequency prediction.

    Args:
        encoder: The encoder to use to obtain the embedding.
        frequency_classifier: The classifier used to move in feature space.
        sample: The sample to retarget.
        frequency: The retarget frequency.

    Returns:
        The retargeted sample in feature space.
    """

    # Transform to feature space.
    feature_space_sample = encoder.predict(sample.reshape(1, *sample.shape))

    # Freeze the classifier model.
    freeze_model(frequency_classifier, True)

    # Make a bridge model to retarget the feature space with gradient descend.
    temp_inp = Input(feature_space_sample.shape[1:])

    # The weights are initally identity so we start at the correct location in feature space.
    temp_dense = Dense(EMBEDDED_SIZE, activation="relu", kernel_initializer="identity")(temp_inp)
    transform_model = Model(temp_inp, temp_dense)

    # Make temporary model to abuse keras gradients.
    temp_out = frequency_classifier(transform_model.output)
    temp_model = Model(temp_inp, temp_out)

    # Search feature space for sample with correct frequency.
    temp_model.compile(loss="mse", optimizer="adam")
    hist = temp_model.fit(
        feature_space_sample, np.array(frequency).reshape(1, 1), epochs=FEATURE_FIT_ITERATIONS, verbose=0
    ).history

    # Check if we failed to converge.
    if (hist["loss"][-1] > 0.01):
        print("Failed to converge. Final loss {0}.".format(hist["loss"][-1]))

    # Unfreeze model.
    freeze_model(frequency_classifier, False)

    # Move feature space sample to different feature space location.
    return transform_model.predict(feature_space_sample)


def train_autoencoder(autoencoder, input_samples, freq_labels):
    """
    Trains a model produced by the build_model function.

    Args:
        autoencoder: The model to train.
        input_samples: Input samples used to train reconstruction on.
        freq_labels: Frequency labels for every sample.
    """

    autoencoder.compile(loss="mse", optimizer="adam")
    autoencoder.fit(input_samples, [input_samples, freq_labels], epochs=TRAINING_EPOCHS)
