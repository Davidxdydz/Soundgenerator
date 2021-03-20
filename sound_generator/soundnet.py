import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    Flatten,
    UpSampling1D,
    Add,
    Lambda
)
from tensorflow.keras.models import Model
from sound_generator.global_configuration import TRAINING_EPOCHS, FEATURE_FIT_ITERATIONS,EMBEDDED_SIZE
import numpy as np


"""
all flat

        samples n   fft-mag n/2     fft-pha n/2 1
layer1:     | a         | b             | b     1
layer2:     | m         | m             | m     1
            -----------------------------
                        |
                       add m                    2
layer3:                 | c                     2
embedded:               | d                     2
                        ---------------------------------
-layer3:                | c                     3       |
                        | m                     3       |
            -----------------------------               |
-layer2:    | a         | b             | b     4       |
-layer1:    | n         | n/2           | n/2   4     classifier
            |           -----------------       4       |
            |                   |               4       |
            |                  ifft             4       |
            |                   |               4       |
            ---------------------               4       |
                    add                         4       |
                mse against samples                 mse against freq

"""
a = 1024
b = 1024
m = 256
c = 128
d = 64



def build_sample_encoder(input_shape):
    """
    TODO
    """
    encoder_input = Input(input_shape)
    encoder_hidden = Dense(a, activation = "tanh")(encoder_input)
    encoder_hidden = Dense(m,activation = "tanh")(encoder_hidden)
    return Model(encoder_input,encoder_hidden,name = "SampleEncoder")

def build_magnitude_encoder(input_shape):
    """
    TODO
    """
    encoder_input = Input(input_shape)
    encoder_hidden = Dense(b, activation = "tanh")(encoder_input)
    encoder_hidden = Dense(m,activation = "tanh")(encoder_hidden)
    return Model(encoder_input,encoder_hidden,name = "MagnitudeEncoder")

def build_phase_encoder(input_shape):
    """
    TODO
    """
    encoder_input = Input(input_shape)
    encoder_hidden = Dense(b, activation = "tanh")(encoder_input)
    encoder_hidden = Dense(m,activation = "tanh")(encoder_hidden)
    return Model(encoder_input,encoder_hidden,name = "PhaseEncoder")

def build_inner_encoder():
    """
    TODO
    """

    sample_encoder_output = Input((m,))
    magnitude_encoder_output = Input((m,))
    phase_encoder_output = Input((m,))
    added = Add()([sample_encoder_output,magnitude_encoder_output,phase_encoder_output])

    encoder_hidden = Dense(c, activation="tanh")(added)
    encoder_out = Dense(d, activation="tanh")(encoder_hidden)

    return Model(inputs = [sample_encoder_output,magnitude_encoder_output,phase_encoder_output], outputs = encoder_out, name="InnerEncoder")


def build_inner_decoder():
    """
    TODO
    """

    decoder_input = Input((d,))
    decoder_hidden = Dense(c,activation="tanh")(decoder_input)
    decoder_out = Dense(m,activation="tanh")(decoder_hidden)

    return Model(decoder_input, decoder_out, name="InnerDecoder")

def build_sample_decoder(sample_shape):
    """
    TODO
    """

    decoder_input = Input((m,))
    decoder_hidden = Dense(a,activation="tanh")(decoder_input)
    decoder_output = Dense(*sample_shape,activation="tanh")(decoder_hidden)
    return Model(decoder_input,decoder_output,name = "SampleDecoder")

def build_magnitude_decoder(magnitude_shape):
    """
    TODO
    """

    decoder_input = Input((m,))
    decoder_hidden = Dense(b,activation="tanh")(decoder_input)
    decoder_output = Dense(*magnitude_shape,activation="tanh")(decoder_hidden)
    return Model(decoder_input,decoder_output,name = "MagnitudeDecoder")

def build_phase_decoder(phase_shape):
    """
    TODO
    """

    decoder_input = Input((m,))
    decoder_hidden = Dense(b,activation="tanh")(decoder_input)
    decoder_output = Dense(*phase_shape,activation="tanh")(decoder_hidden)
    return Model(decoder_input,decoder_output,name = "PhaseDecoder")

def polar_to_complex(inp):
    r = inp[0]
    p = inp[1]
    return tf.cast(r,tf.complex64)* tf.complex(tf.math.cos(p),tf.math.sin(p))


def build_outer_decoder(sample_shape,magnitude_shape,phase_shape):
    """
    TODO
    """

    inner_decoder_output = Input((m,))

    sample_decoder = build_sample_decoder(sample_shape)(inner_decoder_output)
    magnitude_decoder = build_magnitude_decoder(magnitude_shape)(inner_decoder_output)
    phase_decoder = build_phase_decoder(phase_shape)(inner_decoder_output)


    fft_layer = Lambda(polar_to_complex,dtype = 'complex64')([magnitude_decoder,phase_decoder])
    time_layer = Lambda(tf.signal.irfft)(fft_layer)
    adding_layer = Add()([sample_decoder,time_layer])

    return Model(inner_decoder_output,adding_layer,name = "OuterDecoder")

def build_encoder(sample_shape,magnitude_shape,phase_shape):
    """
    TODO
    """
    sample_input = Input(sample_shape)
    magnitude_input = Input(magnitude_shape)
    phase_input = Input(phase_shape)

    sample_encoder = build_sample_encoder(sample_shape)(sample_input)
    magnitude_encoder = build_magnitude_encoder(magnitude_shape)(magnitude_input)
    phase_encoder = build_phase_encoder(phase_shape)(phase_input)

    inner_encoder = build_inner_encoder()
    connected = inner_encoder([sample_encoder,magnitude_encoder,phase_encoder])
    return Model(inputs = [sample_input,magnitude_input,phase_input],outputs = connected, name = "Encoder")

def build_decoder(sample_shape,magnitude_shape,phase_shape):
    """
    TODO
    """
    feature_space = Input((d,))

    inner_decoder = build_inner_decoder()(feature_space)

    outer_decoder = build_outer_decoder(sample_shape,magnitude_shape,phase_shape)(inner_decoder)

    return Model(feature_space,outer_decoder,name = "Decoder")


def build_freq_classifier():
    """
    Builds a classifier model to investigate frequencies in embedded space.

    Args:
        feature_shape: Tuple containing the shape of the embedded input space.

    Returns:
        A single output classifier that estimates the frequency a specific embedded
        point represents.
    """

    freq_input = Input((d,))
    freq_classifier = Dense(64, activation="tanh")(freq_input)
    freq_classifier = Dense(16, activation="tanh")(freq_classifier)
    freq_classifier = Dense(1, activation="tanh")(freq_classifier)
    return Model(freq_input, freq_classifier, name="FreqClassifier")

def build_autoencoder(sample_shape,magnitude_shape,phase_shape):
    sample_input = Input(sample_shape)
    magnitude_input = Input(magnitude_shape)
    phase_input = Input(phase_shape)

    encoder = build_encoder(sample_shape,magnitude_shape,phase_shape)([sample_input,magnitude_input,phase_input])
    decoder = build_decoder(sample_shape,magnitude_shape,phase_shape)(encoder)
    freq_classifier = build_freq_classifier()(encoder)

    return Model(inputs = [sample_input,magnitude_input,phase_input],outputs = [decoder,freq_classifier],name = "Autoencoder")

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
    temp_dense = Dense(EMBEDDED_SIZE, activation="tanh", kernel_initializer="identity")(temp_inp)
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


def train_autoencoder(autoencoder, samples,magnitudes,phases, freq_labels):
    """
    Trains a model produced by the build_model function.

    Args:
        autoencoder: The model to train.
        input_samples: Input samples used to train reconstruction on.
        freq_labels: Frequency labels for every sample.
    """

    autoencoder.compile(loss="mse", optimizer="adam")
    autoencoder.fit([samples,magnitudes,phases], [samples, freq_labels], epochs=TRAINING_EPOCHS)
