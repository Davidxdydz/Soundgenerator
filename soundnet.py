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

    # (n,)
    encoder_input = Input(input_shape)
    # (n, 16)
    encoder_hidden = Conv1D(32, 8, padding="same", activation="tanh")(encoder_input)
    # (n / 4, 16)
    encoder_hidden = MaxPool1D(4, padding="same")(encoder_hidden)
    # (n / 4, 32)
    encoder_hidden = Conv1D(64, 8, padding="same", activation="tanh")(encoder_hidden)
    # (n / 16, 32)
    encoder_hidden = MaxPool1D(4, padding="same")(encoder_hidden)

    # TODO make this constant maybe so that input shape is not correlated with embedded_size
    # (n / 16, embedded_size)
    encoder_out = Dense(embedded_size, activation="tanh")(encoder_hidden)

    return Model(encoder_input, encoder_out,name = "Encoder")


def build_decoder(feature_shape):

    decoder_input = Input(feature_shape)
    decoder_hidden = Conv1D(32, 8, padding="same", activation="tanh")(decoder_input)
    decoder_hidden = UpSampling1D(4)(decoder_input)
    decoder_hidden = Conv1D(16, 8, padding="same", activation="tanh")(decoder_hidden)
    decoder_hidden = UpSampling1D(4)(decoder_hidden)
    decoder_out = Conv1D(1, 8, padding="same", activation="tanh")(decoder_hidden)
    return Model(decoder_input, decoder_out,name="Decoder")


def build_freq_classifier(feature_shape):
    freq_input = Input(feature_shape)
    freq_flatten = Flatten()(freq_input)
    freq_classifier = Dense(8, activation="tanh")(freq_flatten)
    freq_classifier = Dense(1, activation="tanh")(freq_classifier)
    return Model(freq_input, freq_classifier,name = "FreqClassifier")


def build_model(input_shape, embedded_size):

    encoder = build_encoder(input_shape, embedded_size)
    decoder = build_decoder(encoder.output_shape[1:])
    freq_classifier = build_freq_classifier(encoder.output_shape[1:])

    autoencoder_input = Input(input_shape)
    autoencoder_enc = encoder(autoencoder_input)
    autoencoder_freq = freq_classifier(autoencoder_enc)
    autoencoder_dec = decoder(autoencoder_enc)

    return (
        Model(autoencoder_input, [autoencoder_dec, autoencoder_freq],name = "autoencoder"),
        encoder,
        decoder,
    )


def train_autoencoder(autoencoder, input_samples, freq_labels):

    autoencoder.compile(loss="mse", optimizer="adam")
    autoencoder.fit(input_samples, [input_samples, freq_labels], epochs = 1000)
