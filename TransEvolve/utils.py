import numpy as np
import tensorflow as tf
import math

def positional_embedding(length,
                          hidden_size,
                          min_timescale=1.0,
                          max_timescale=1.0e4):
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal[tf.newaxis, ...]

def create_padding_mask(seq):
  seq = tf.math.equal(seq, 0)
  seq = seq[:, tf.newaxis, tf.newaxis, :]
  seq = tf.repeat(seq, repeats=tf.shape(seq)[-1], axis=-2)
  seq_t = tf.transpose(seq, perm=(0, 1, 3, 2))
  return tf.cast(tf.math.logical_or(seq, seq_t), dtype=tf.float32)

def create_look_ahead_mask(maxlen):
  mask = 1 - tf.linalg.band_part(tf.ones((maxlen, maxlen)), -1, 0)
  return mask

def create_combined_mask(tar, src):
  tar = tf.math.equal(tar, 0)
  tar = tar[:, tf.newaxis, tf.newaxis, :]
  tar = tf.repeat(tar, repeats=tf.shape(src)[-1], axis=-2)
  src = tf.math.equal(src, 0)
  src = src[:, tf.newaxis, tf.newaxis, :]
  src = tf.repeat(src, repeats=tf.shape(tar)[-1], axis=-2)
  src = tf.transpose(src, perm=(0, 1, 3, 2))
  return tf.transpose(tf.cast(tf.math.logical_or(tar, src), dtype=tf.float32), perm=(0, 1, 3, 2))

def create_padding_array(seq):
  seq = tf.math.equal(seq, 0)
  seq = seq[:, tf.newaxis, tf.newaxis, :]
  return tf.cast(seq, dtype=tf.float32)

def create_encoder_decoder_mask_from_array(tar, enc_pad):
  tar = tf.math.equal(tar, 0)
  tar = tar[:, tf.newaxis, tf.newaxis, :]
  tar = tf.repeat(tar, repeats=tf.shape(enc_pad)[-1], axis=-2)
  src = tf.repeat(tf.math.not_equal(enc_pad, 0.), repeats=tf.shape(tar)[-1], axis=-2)
  src = tf.transpose(src, perm=(0, 1, 3, 2))
  return tf.transpose(tf.cast(tf.math.logical_or(tar, src), dtype=tf.float32), perm=(0, 1, 3, 2))

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate = 0.5, 
               hidden_size = 512, 
               warmup_steps = 16000):
  
    super(LearningRateSchedule, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.hidden_size = hidden_size
    self.warmup_steps = tf.cast(warmup_steps, tf.float32)

  def __call__(self, global_step):

    global_step = tf.cast(global_step, tf.float32)
    learning_rate = self.initial_learning_rate
    learning_rate *= (self.hidden_size**-0.5)
    learning_rate *= tf.minimum(1.0, global_step / self.warmup_steps)
    learning_rate /= tf.sqrt(tf.maximum(global_step, self.warmup_steps))
    return learning_rate

  def get_config(self):

    return {
        'initial_learning_rate': self.initial_learning_rate,
        'hidden_size': self.hidden_size,
        'warmup_steps': self.warmup_steps,
    }

def padded_cross_entropy_loss(logits, labels, smoothing=0.1):
  vocab_size = tf.shape(logits)[-1]
  confidence = 1.0 - smoothing
  low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
  soft_targets = tf.one_hot(
      tf.cast(labels, tf.int32),
      depth=vocab_size,
      on_value=confidence,
      off_value=low_confidence)
  xentropy = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=soft_targets)

  normalizing_constant = -(
      confidence * tf.math.log(confidence) +
      tf.cast(vocab_size - 1, tf.float32) * low_confidence *
      tf.math.log(low_confidence + 1e-20))
  xentropy -= normalizing_constant

  weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
  return tf.reduce_sum(xentropy * weights)/tf.reduce_sum(weights)
