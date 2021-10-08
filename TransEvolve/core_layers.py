import tensorflow as tf
from tensorflow.keras import layers, activations, backend
import numpy as np
import math

def _glorot_initializer(fan_in, fan_out):
  limit = math.sqrt(6.0 / (fan_in + fan_out))
  return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

def _time_signal(coeffs, step):
  x = tf.cast(step, dtype=tf.float32)*(math.pi/64.)
  p = tf.cast(tf.range(tf.shape(coeffs)[0]//2), dtype=tf.float32)
  y = tf.concat([tf.math.sin(x*p), tf.math.cos(x*p)], axis=0)
  return y*coeffs

class Layer1(layers.Layer):

  def __init__(self, 
               hidden_dim=64, 
               time_dim=64,
               num_head=8, 
               dropout=0.1,
               **kwargs):
    super(Layer1, self).__init__()

    self.hidden_dim = hidden_dim
    self.time_dim = time_dim
    self.num_head = num_head
    self.qdropout = tf.keras.layers.Dropout(rate=dropout)
    self.kdropout = tf.keras.layers.Dropout(rate=dropout)

    self.WQ = self.add_weight(shape=(hidden_dim+num_head*time_dim, 
                                     hidden_dim+num_head*time_dim),
                             initializer=_glorot_initializer(hidden_dim+num_head*time_dim, 
                                                             hidden_dim+num_head*time_dim),
                             trainable=True, name='query_projection')
    self.WK = self.add_weight(shape=(hidden_dim+num_head*time_dim, 
                                     hidden_dim+num_head*time_dim),
                             initializer=_glorot_initializer(hidden_dim+num_head*time_dim, 
                                                             hidden_dim+num_head*time_dim),
                             trainable=True, name='key_projection')
    
  def call(self, Iq, Ik, training):

    Q = self.qdropout(tf.einsum('ab, ...lb -> ...la', self.WQ[:, :self.hidden_dim], Iq), training=training)
    K = self.qdropout(tf.einsum('ab, ...lb -> ...la', self.WK[:, :self.hidden_dim], Ik), training=training)
    Q0 = tf.reshape(Q, [-1, tf.shape(Q)[1], self.num_head, tf.shape(Q)[-1]//self.num_head])
    K0 = tf.reshape(K, [-1, tf.shape(K)[1], self.num_head, tf.shape(K)[-1]//self.num_head])
    A0 = tf.einsum('...shd,...lhd->...hsl', Q0, K0)
    A0 = A0/tf.sqrt(tf.cast(tf.shape(Q0)[-1], dtype=tf.float32))
    
    _WQ = tf.reshape(self.WQ[:, self.hidden_dim:], [-1, self.num_head, self.time_dim])
    _WK = tf.reshape(self.WK[:, self.hidden_dim:], [-1, self.num_head, self.time_dim])
    A2 = tf.einsum('dnt, dnt -> nt', _WQ, _WK)[tf.newaxis, :, tf.newaxis, :]

    A1K = tf.einsum('dnt, ...ld -> ...nlt', _WQ, K)
    A1Q = tf.einsum('dnt, ...ld -> ...nlt', _WK, Q)
      
    return A0, A1Q, A1K, A2 

  def compute_mask(self, inputs, mask=None):
    return mask
    
  def get_config(self):

    config = super().get_config().copy()
    config.update({
          'hidden_dim': self.hidden_dim,
          'num_head': self.num_head,
        })
    return config

class Layer2(layers.Layer):
    
  def __init__(self, 
               hidden_dim=64, 
               num_head=8, 
               dropout=0.1,
               **kwargs):
    super(Layer2, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_head = num_head
    self.dropout = tf.keras.layers.Dropout(rate=dropout)
    self.norm_layer = tf.keras.layers.LayerNormalization()
    
  def build(self, input_shape): 
    self.coeffs = self.add_weight(shape=(self.hidden_dim,),
                             initializer=_glorot_initializer(self.hidden_dim, 0),
                             trainable=True, name='coefficients')
    self.WO = self.add_weight(shape=(self.hidden_dim, self.hidden_dim),
                             initializer=_glorot_initializer(self.hidden_dim, self.hidden_dim),
                             trainable=True, name='linear_out')

  def call(self, A0, A1Q, A1K, A2, V, i, mask, training):

    h = self.norm_layer(V)
    t = tf.reshape(_time_signal(self.coeffs, i), [self.num_head, self.hidden_dim//self.num_head])
    A1 = tf.einsum('...nlt, nt -> ...nl', A1Q, t)[:, :, :, tf.newaxis] + tf.einsum('...nlt, nt -> ...nl', A1K, t)[:, :, tf.newaxis, :]
    A2 = tf.einsum('anlt, nt -> anl', A2, t*t)[:, :, :, tf.newaxis]
    A = A0 + A1 + A2
    A += mask*-1e9
    A = tf.nn.softmax(A, axis=-1)
    h = tf.reshape(h, [-1, tf.shape(h)[1], self.num_head, self.hidden_dim//self.num_head])
    h = tf.einsum('...nlk, ...knd-> ...lnd', A, h)
    h = tf.reshape(h, [-1, tf.shape(h)[1], self.hidden_dim])
    h = tf.einsum('...a, ab-> ...b', h, self.WO)
    return self.dropout(h, training=training)
    
  def compute_mask(self, inputs, mask=None):
    return mask

  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'num_head': self.num_head,
          'hidden_dim': self.hidden_dim,
        })
    return config

class TemporalProjection(layers.Layer):
    
  def __init__(self, 
               in_dim=512, 
               out_dim=2048, 
               dropout=0.1,
               **kwargs):
    super(TemporalProjection, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.coeffs = tf.random.normal([in_dim, in_dim//2], 0., in_dim, tf.float32, seed=1)
    self.phases = tf.random.normal([out_dim, out_dim//2], 0., out_dim, tf.float32, seed=1)

    self.svals = self.add_weight(shape=(in_dim, 1),
                             initializer=_glorot_initializer(min(in_dim, out_dim), 0),
                             trainable=True, name='svals')
    self.bias = self.add_weight(shape=(out_dim,),
                             initializer=tf.keras.initializers.Zeros(),
                             trainable=True, name='bias')

  def call(self, h, i):
    x1 = tf.cast(i, dtype=tf.float32)*(math.pi/self.in_dim)
    x2 = tf.cast(i, dtype=tf.float32)*(math.pi/self.out_dim)
    p = tf.cast(tf.range(self.in_dim//2), dtype=tf.float32)[tf.newaxis, :]
    q = tf.cast(tf.range(self.out_dim//2), dtype=tf.float32)[tf.newaxis, :]
    U = tf.concat([tf.math.sin((x1*p)*self.coeffs), 
                    tf.math.cos((x1*p)*self.coeffs)], 
                  axis=1)/tf.sqrt(tf.cast(self.out_dim, dtype=tf.float32))
    V = tf.concat([tf.math.sin((x2*q)*self.phases), 
                    tf.math.cos((x2*q)*self.phases)], 
                  axis=1)/tf.sqrt(tf.cast(self.out_dim, dtype=tf.float32))
    S = tf.eye(self.in_dim, num_columns=self.out_dim)*self.svals
    W = tf.einsum('ab, bc -> ac', tf.einsum('ab, bc -> ac', U, S), V)
    h = tf.einsum('...ab, bc -> ...ac', h, W) + self.bias
    return h
    
  def compute_mask(self, inputs, mask=None):
    return mask

  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'in_dim': self.in_dim,
          'out_dim': self.out_dim,
        })
    return config

class Layer3(layers.Layer):
    
  def __init__(self, 
               projection_type='random', 
               hidden_dim=64, 
               num_head=8, 
               dropout=0.1,
               **kwargs):
    super(Layer3, self).__init__()
    self.dropout = tf.keras.layers.Dropout(rate=dropout)
    self.norm_layer = tf.keras.layers.LayerNormalization()
    self.projection_type = projection_type
    if projection_type=='full':
      self.linear_1 = tf.keras.layers.Dense(4*hidden_dim)
      self.linear_2 = tf.keras.layers.Dense(hidden_dim)
    elif projection_type=='random':
      self.linear_1 = TemporalProjection(out_dim=4*hidden_dim, in_dim=hidden_dim)
      self.linear_2 = TemporalProjection(in_dim=4*hidden_dim, out_dim=hidden_dim)
    else:
      raise TypeError

  def call(self, h, i, training):

    h = self.norm_layer(h)
    if self.projection_type=='full':
      h = self.linear_2(tf.nn.relu(self.linear_1(h)))
    else:
      h = self.linear_2(tf.nn.relu(self.linear_1(h, i)), i)
    return self.dropout(h, training=training)
    
  def compute_mask(self, inputs, mask=None):
    return mask

  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'hidden_dim': self.hidden_dim,
        })
    return config


"""#Embedding Layer"""

class EmbeddingSharedWeights(tf.keras.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_dim):
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_dim

  def build(self, input_shape):
    self.shared_weights = self.add_weight("weights",
                                          shape=[self.vocab_size, self.hidden_size],
                                          initializer=tf.random_normal_initializer(
                                            mean=0., stddev=self.hidden_size**-0.5))
    super(EmbeddingSharedWeights, self).build(input_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
          "vocab_size": self.vocab_size,
          "hidden_dim": self.hidden_size,
        })
    return config
    
  def call(self, inputs, mode="embedding"):
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == "linear":
      return self._linear(inputs)
    else:
      raise ValueError("mode {} is not valid.".format(mode))

  def _embedding(self, inputs):
      # Create binary mask of size [batch_size, length]
    inputs = tf.cast(inputs, tf.int32)
    embeddings = tf.gather(self.shared_weights, inputs)
    mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
    embeddings *= tf.expand_dims(mask, -1)
    # Scale embedding by the sqrt of the hidden size
    embeddings *= self.hidden_size ** 0.5

    return embeddings

  def _linear(self, inputs):
    batch_size = tf.shape(inputs)[0]
    length = tf.shape(inputs)[1]

    x = tf.reshape(inputs, [-1, self.hidden_size])
    logits = tf.matmul(x, self.shared_weights, transpose_b=True)

    return tf.reshape(logits, [batch_size, length, self.vocab_size]) 
