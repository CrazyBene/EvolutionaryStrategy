import keras
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
import utils
import numpy as np

class NeuralNetwork:

    def __init__(self, input_shape=None, output_dim=None, model=None, img_number=4):
        if model == None:
            self.img_number = img_number
            self.model = self._build_model((input_shape[0], input_shape[1], img_number), output_dim)
            self.obs_list = np.zeros((1, input_shape[0], input_shape[1], self.img_number))
        else:
            self.model = model
            input_shape = self.model.input.shape.as_list()
            self.img_number = input_shape[3]
            self.obs_list = np.zeros((1, input_shape[1], input_shape[2], self.img_number))
            
    def copy(self):
        return NeuralNetwork(model=self.model)        

    def _build_model(self, input_shape, output_dim):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=input_shape))
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(output_dim))
        return model

    def predict(self, obs):
        obs = self._convert_obs(obs)
        self._add_obs(obs)
        output = self.model.predict(self.obs_list)
        return output

    def set_weights(self, weights):
        self.model.set_weights(weights)
        self._reset_obs()

    def get_weights(self):
        return self.model.get_weights()
    
    def _convert_obs(self, obs):
        obs = utils.rgb2gray(obs)
        obs = utils.rescale_image(obs)
        obs = utils.normalize_image(obs)
        return obs

    def _add_obs(self, obs):
        for i in range(0, self.img_number-1):
            self.obs_list[..., i] = self.obs_list[..., i+1]

        self.obs_list[..., self.img_number-1] = obs

    def _reset_obs(self):
        self.obs_list = np.zeros(self.obs_list.shape)

    def save(self, path):
        self.model.save(path, include_optimizer=False)

    @staticmethod
    def load(path):
        model = load_model(path, compile=False)
        nn = NeuralNetwork(model=model)
        return nn

    def summary(self):
        self.model.summary()