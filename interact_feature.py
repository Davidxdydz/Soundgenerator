from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import numpy as np


def build_model():

    encoder_inp = Input((28 ** 2,))
    encoder_dense = Dense(64, activation="sigmoid")(encoder_inp)
    encoder_dense = Dense(32, activation="sigmoid")(encoder_inp)
    encoder_out = Dense(2, activation="sigmoid")(encoder_dense)

    decoder_inp = Input((2,))
    decoder_dense = Dense(32, activation="sigmoid")(decoder_inp)
    decoder_dense = Dense(64, activation="sigmoid")(decoder_inp)
    decoder_out = Dense(28 ** 2, activation="sigmoid")(decoder_dense)

    encoder = Model(encoder_inp, encoder_out)
    decoder = Model(decoder_inp, decoder_out)

    aa_inp = Input((28 ** 2,))
    aa_enc = encoder(aa_inp)
    aa_dec = decoder(aa_enc)

    aa = Model(aa_inp, aa_dec)

    return aa, encoder, decoder


def load_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.divide(x_train, 255.0).reshape((-1, 28 ** 2))
    x_test = np.divide(x_test, 255.0).reshape((-1, 28 ** 2))

    return np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])


def train_model(aa, x_train):

    aa.compile(loss="mse", optimizer="adam")
    aa.fit(x_train, x_train, epochs=10)


if __name__ == "__main__":

    aa, encoder, decoder = build_model()
    x_train, y_train = load_data()
    train_model(aa, x_train)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    def image_from_feature(event):
        if not (event.xdata and event.ydata):
            return
        data = np.array([event.xdata, event.ydata]).reshape(1, 2)
        img = decoder.predict(data)
        ax[1].imshow(img.reshape(28, 28))
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", image_from_feature)

    embedded = encoder.predict(x_train)

    scatter = ax[0].scatter(embedded[:, 0], embedded[:, 1], c=y_train, s=2)
    ax[0].set_title("Feature space (Click to explore)")
    legend1 = ax[0].legend(
        *scatter.legend_elements(),
        loc="lower left",
        title="Classes",
        bbox_to_anchor=(1, 0),
    )

    ax[0].add_artist(legend1)
    ax[1].axis("off")
    plt.show()