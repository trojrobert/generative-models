from keras.models import Model
from keras.layers import Input, Dense 
from keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt 

class AutoEncoder:

  def __init__(self, encoding_dim=64):
    self.encoding_dim = encoding_dim
    x_train, x_test = self.get_data()
    self.x_train = x_train
    self.x_test = x_test
    self.n, self.d = x_train.shape
    print(self.x_train[0].shape)
    
  def _encoder(self):
    input_imgs = Input(shape=(self.x_train[0].shape))
    encoded = Dense(self.encoding_dim, activation='relu')(input_imgs)
    encoder_model = Model(input_imgs, encoded)
    self.encoder_model = encoder_model
    return encoder_model

  def _decoder(self):
    encoded_input = Input(shape=(encoding_dim, ))
    decoded = Dense(784, activation='sigmoid')(encoded_input)
    decoder_model = Model(encoded_input, decoded)
    self.decoder_model = decoder_model
    return decoder_model


  def encoder_decoder(self):
    ec = self._encoder()
    dc = self._decoder()

    inputs = Input(shape=self.x_train[0].shape)
    ec_output = ec(inputs)

    dc_output = dc(ec_output)
    autoencoder = Model(inputs, dc_output)

    self.autoencoder = autoencoder
    return autoencoder
  
  def fit(self, epochs=50, batch_size=256 ):
    self.autoencoder.compile(optimizer='sgd', loss = 'binary_crossentropy')
    
    autoencoder.fit(self.x_train, self.x_train, 
                    epochs=epochs, 
                    batch_size = batch_size, 
                    shuffle=True,
                    validation_data=(self.x_test, self.x_test))



  def get_data(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')/255
    y_train = y_train.astype('float32')/255

    x_train = x_train.reshape((len(x_train), (np.prod(x_train.shape[1:]))))
    x_test = x_test.reshape((len(x_test), (np.prod(x_train.shape[1:]))))

    return x_train, x_test

  def save(self):

    if not os.path.exists(r'./weights'):
      os.mkdir(r'./weights')
    else:
      self.encoder_model.save(r'./weights/encoder_weights.h5')
      self.decoder_model.save(r'./weights/decoder_weights.h5')
      self.autoencoder.save(r'./weights/autoencoder_weights.h5')


if __name__ == '__main__':
  set_seeds(2)
  ae = AutoEncoder(encoding_dim=64)
  ae.encoder_decoder()
  ae.fit(epochs=50, batch_size=256)
  ae.save()