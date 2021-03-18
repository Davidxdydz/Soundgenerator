from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    Flatten,
    MaxPool1D,
    UpSampling1D,
)
from tensorflow.keras.models import Model


def build_encoder(input_shape, embedded_size):
    """
    Builds an encoder model for use in an autoencoder.

    Args:
        input_shape: Tuple with network input shape
        embedded_size: Size of the final encoder embedding layer.

    Returns:
        Encoder model with specified dimensions.
    """

    # TODO increase params, atm only 26k params
    # (n,)
    encoder_input = Input(input_shape)
    # (n, 16)
    encoder_hidden = Conv1D(64, 8, padding="same", activation="relu")(encoder_input)
    # (n / 4, 16)
    encoder_hidden = MaxPool1D(4, padding="same")(encoder_hidden)
    # (n / 4, 32)
    encoder_hidden = Conv1D(32, 8, padding="same", activation="relu")(encoder_hidden)
    # (n / 16, 32)
    encoder_hidden = MaxPool1D(4, padding="same")(encoder_hidden)

    # TODO make this constant maybe so that input shape is not correlated with embedded_size
    # (n / 16, embedded_size)
    encoder_out = Dense(embedded_size, activation="relu")(encoder_hidden)

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
    decoder_hidden = Conv1D(32, 8, padding="same", activation="relu")(decoder_input)
    decoder_hidden = UpSampling1D(4)(decoder_input)
    decoder_hidden = Conv1D(64, 8, padding="same", activation="relu")(decoder_hidden)
    decoder_hidden = UpSampling1D(4)(decoder_hidden)
    decoder_out = Conv1D(1, 8, padding="same", activation="relu")(decoder_hidden)
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
    freq_classifier = Dense(8, activation="relu")(freq_flatten)
    freq_classifier = Dense(1, activation="relu")(freq_classifier)
    return Model(freq_input, freq_classifier, name="FreqClassifier")


def build_model(input_shape, embedded_size):
    """
    Builds a frequency autoencoder model with a side output for
    that estimates the frequncy of the input sample.

    Args:
        input_shape: Tuple with network input and output shape
        embedded_size: Size of the final encoder embedding layer.

    Returns:
        A triple containing the complete autoencoder, the encoder and 
        decoder.
    """

    encoder = build_encoder(input_shape, embedded_size)
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
    )


def train_autoencoder(autoencoder, input_samples, freq_labels):
    """
    Trains a model produced by the build_model function.

    Args:
        autoencoder: The model to train.
        input_samples: Input samples used to train reconstruction on.
        freq_labels: Frequency labels for every sample.
    """

    autoencoder.compile(loss="mse", optimizer="adam")
    autoencoder.fit(input_samples, [input_samples, freq_labels], epochs=1000)
