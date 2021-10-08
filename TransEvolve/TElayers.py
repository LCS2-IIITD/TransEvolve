import tensorflow as tf
from TransEvolve.TransEvolve import core_layers

class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self,  
               hidden_dim=512, 
               num_head=8,
               projection_type='random',
               dropout=0.1,
               **kwargs):
    super(EncoderLayer, self).__init__()
    self.apply_attention = core_layers.Layer2(hidden_dim=hidden_dim, 
                                  num_head=num_head, 
                                  dropout=dropout)
    self.apply_projection = core_layers.Layer3(projection_type=projection_type,
                                   hidden_dim=hidden_dim, 
                                   dropout=dropout)
    
  def call(self, A, x, i, mask, training):
    A0, A1Q, A1K, A2 = A
    x += self.apply_attention(A0, A1Q, A1K, A2, x, i, mask, training)
    x += self.apply_projection(x, i, training)
    return x
  
  def get_config(self):
    config = super().get_config().copy()
    return config

class EncoderBlock(tf.keras.layers.Layer):

  def __init__(self,  
               hidden_dim=512, 
               projection_type='random',
               num_head=8,
               num_blocks=6, 
               dropout=0.1,
               **kwargs):
    super(EncoderBlock, self).__init__()
    self.num_blocks = num_blocks
    self.compute_attention = core_layers.Layer1(hidden_dim=hidden_dim, 
                                    num_head=num_head, 
                                    dropout=dropout)
    self.encoder = [EncoderLayer(hidden_dim=hidden_dim, 
                                 num_head=num_head,
                                 projection_type=projection_type, 
                                 dropout=dropout) for _ in range(num_blocks)]
    self.output_normalization = tf.keras.layers.LayerNormalization()
    
  def call(self, x, mask, training):
    A = self.compute_attention(x, x, training)
    for i in range(self.num_blocks):
      x = self.encoder[i](A, x, i, mask, training)
    return self.output_normalization(x)
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'num_blocks': self.num_blocks,
        })
    return config

class Encoder(tf.keras.layers.Layer):

  def __init__(self,  
               hidden_dim=512, 
               projection_type='random',
               num_head=8,
               num_blocks=3,
               num_encoder=2, 
               dropout=0.1,
               **kwargs):
    super(Encoder, self).__init__()
    self.num_encoder = num_encoder
    self.encoder = [EncoderBlock(hidden_dim=hidden_dim, 
                                 num_head=num_head,
                                 projection_type=projection_type,
                                 num_blocks=num_blocks, 
                                 dropout=dropout) for _ in range(num_encoder)]

  def call(self, x, mask, training):
    for i in range(self.num_encoder):
      x = self.encoder[i](x, mask, training)
    return x
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'num_encoder': self.num_encoder,
        })
    return config

"""#Decoder definition"""

class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self,  
               hidden_dim=512, 
               projection_type='random',
               num_head=8, 
               dropout=0.1,
               **kwargs):
    super(DecoderLayer, self).__init__()
    self.apply_decoder_attention = core_layers.Layer2(hidden_dim=hidden_dim, 
                                          num_head=num_head, 
                                          dropout=dropout)
    self.apply_encdec_attention = core_layers.Layer2(hidden_dim=hidden_dim, 
                                          num_head=num_head, 
                                          dropout=dropout)
    self.apply_projection = core_layers.Layer3(hidden_dim=hidden_dim, 
                                   projection_type=projection_type,
                                   dropout=dropout)
    
  def call(self, A, A_cross, x, enc, i, d_mask, ed_mask, training):
    A0, A1Q, A1K, A2 = A
    x += self.apply_decoder_attention(A0, A1Q, A1K, A2, x, i, d_mask, training)
    A0, A1Q, A1K, A2 = A_cross
    x += self.apply_encdec_attention(A0, A1Q, A1K, A2, enc, i, ed_mask, training)
    x += self.apply_projection(x, i, training)
    return x
  
  def get_config(self):
    config = super().get_config().copy()
    return config

class DecoderBlock(tf.keras.layers.Layer):

  def __init__(self,  
               hidden_dim=512, 
               num_head=8,
               projection_type='random',
               num_blocks=3, 
               dropout=0.1,
               **kwargs):
    super(DecoderBlock, self).__init__()
    self.num_blocks = num_blocks
    self.compute_decoder_attention = core_layers.Layer1(hidden_dim=hidden_dim, 
                                            num_head=num_head, 
                                            dropout=dropout)
    self.compute_encdec_attention = core_layers.Layer1(hidden_dim=hidden_dim, 
                                           num_head=num_head, 
                                           dropout=dropout)
    self.decoder = [DecoderLayer(hidden_dim=hidden_dim, 
                                 num_head=num_head,
                                 projection_type=projection_type, 
                                 dropout=dropout) for _ in range(num_blocks)]
    self.output_normalization = tf.keras.layers.LayerNormalization()
    
  def call(self, x, enc, d_mask, ed_mask, training):
    A = self.compute_decoder_attention(x, x, training)
    A_cross = self.compute_encdec_attention(x, enc, training)
    for i in range(self.num_blocks):
      x = self.decoder[i](A, A_cross, x, enc, i, d_mask, ed_mask, training)
    return self.output_normalization(x)
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'num_blocks': self.num_blocks,
        })
    return config

class Decoder(tf.keras.layers.Layer):

  def __init__(self,  
               hidden_dim=512, 
               num_head=8,
               projection_type='random',
               num_blocks=3,
               num_decoder=2, 
               dropout=0.1,
               **kwargs):
    super(Decoder, self).__init__()
    self.num_decoder = num_decoder
    self.decoder = [DecoderBlock(hidden_dim=hidden_dim, 
                                 num_head=num_head, 
                                 num_blocks=num_blocks,
                                 projection_type=projection_type,
                                 dropout=dropout) for _ in range(num_decoder)]
    
  def call(self, x, enc, d_mask, ed_mask, training):
    for i in range(self.num_decoder):
      x = self.decoder[i](x, enc, d_mask, ed_mask, training)
    return x
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
          'num_decoder': self.num_decoder,
        })
    return config
