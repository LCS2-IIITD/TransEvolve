import sys
import tensorflow as tf
from TransEvolve.TransEvolve import utils
from TransEvolve.TransEvolve import TElayers
from TransEvolve.TransEvolve import core_layers
from TransEvolve.TransEvolve import beam_search
from TransEvolve.TransEvolve import bleu_hook

class EncoderDecoderModel(tf.keras.models.Model):
  def __init__(self,
               input_dim=32000,
               hidden_dim=512,
               num_head=8,
               projection_type='full',
               num_layers_per_block=3,
               num_blocks=2,
               dropout=0.1,
               **kwargs):
    super(EncoderDecoderModel, self).__init__()
    self.tiedEmbedding = core_layers.EmbeddingSharedWeights(vocab_size=input_dim, 
                                                hidden_dim=hidden_dim)
    self.pos_encoding = utils.positional_encoding(10000, 
                                            hidden_dim)
    self.encoder = TElayers.Encoder(hidden_dim=hidden_dim,
                           num_head=num_head,
                           projection_type=projection_type,
                           num_blocks=num_layers_per_block,
                           num_encoder=num_blocks,
                           dropout = dropout)
    self.decoder = TElayers.Decoder(hidden_dim=hidden_dim,
                           num_head=num_head,
                           projection_type=projection_type,
                           num_blocks=num_layers_per_block,
                           num_decoder=num_blocks,
                           dropout = dropout)
    
  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, src_inp, tar_inp, training):

    src_maxlen = tf.shape(src_inp)[1]
    tar_maxlen = tf.shape(tar_inp)[1]
    enc_mask = utils.create_padding_mask(src_inp)
    enc_in = self.tiedEmbedding(src_inp) + self.pos_encoding[:, :src_maxlen, :]
    v_enc = self.encoder(enc_in, enc_mask, training)

    dec_mask = tf.maximum(utils.create_padding_mask(tar_inp), 
                          utils.create_look_ahead_mask(tar_maxlen))
    ed_mask = utils.create_combined_mask(tar_inp, src_inp)
    dec_in = self.tiedEmbedding(tar_inp) + self.pos_encoding[:, :tar_maxlen, :]
    dec_out = self.decoder(dec_in, v_enc, dec_mask, ed_mask, training)

    logits = self.tiedEmbedding(dec_out, mode='linear')
    return logits

  def encode(self, src_inp):
    src_maxlen = tf.shape(src_inp)[1]
    enc_mask = utils.create_padding_mask(src_inp)
    enc_in = self.tiedEmbedding(src_inp) + self.pos_encoding[:, :src_maxlen, :]
    v_enc = self.encoder(enc_in, enc_mask, False)
    return v_enc

  def decode(self, enc_out, enc_pad_array, tar_inp):
    tar_maxlen = tf.shape(tar_inp)[1]
    dec_mask = tf.maximum(utils.create_padding_mask(tar_inp), 
                          utils.create_look_ahead_mask(tar_maxlen))
    ed_mask = utils.create_encoder_decoder_mask_from_array(tar_inp, enc_pad_array)

    dec_in = self.tiedEmbedding(tar_inp) + self.pos_encoding[:, :tar_maxlen, :]
    dec_out = self.decoder(dec_in, enc_out, dec_mask, ed_mask, False)
    logits = self.tiedEmbedding(dec_out, mode='linear')
    return logits

  def batch_evaluate(self, dataset, 
                     vocab_size, 
                     tokenizer, 
                     resultsFileName, 
                     alpha=0.6, 
                     beam_size=4,
                     extra_decode_len=50):
    @tf.function(input_signature=[tf.TensorSpec([None, None,], tf.int32), tf.TensorSpec([], tf.int32)])
    def predict_step(src_inp, max_decode_len):
      e_out = self.encode(src_inp)

      initial_ids = tf.ones([tf.shape(e_out)[0]], dtype=tf.int32)

      def symbols_to_logits_fn(ids, i, states):
        enc_out = states["encoder_output"]
        enc_pad = states["enc_pad"]
        dec_out = self.decode(enc_out, enc_pad, ids)
        dec_out = dec_out[:,-1,:]
        return dec_out, states

      input_cache={}
      input_cache["encoder_output"] = e_out
      input_cache["enc_pad"] = utils.create_padding_array(src_inp)

      decoded_ids, _ = beam_search.sequence_beam_search(
              symbols_to_logits_fn=symbols_to_logits_fn,
              initial_ids=initial_ids,
              initial_cache=input_cache,
              vocab_size=vocab_size,
              beam_size=beam_size,
              alpha=alpha,
              max_decode_length=max_decode_len,
              eos_id=2,
              padded_decode=False)
      return decoded_ids[:, 0, :]
    hyp_fname = "_".join([resultsFileName, str(alpha), str(beam_size), "hyp.txt"])
    ref_fname = "_".join([resultsFileName, str(alpha), str(beam_size), "ref.txt"])
    file_outputs = open(hyp_fname, "w+")
    file_gold_labels = open(ref_fname, "w+")

    predictions, references = [], []
    for inp, ref in dataset:
      max_decode_len = int(extra_decode_len + inp.shape[1])
      predictions.append(predict_step(inp, max_decode_len))
      references.append(ref)
      
    for pred_batch, ref_batch in zip(predictions, references):
      for pred, ref in zip(tf.unstack(pred_batch), tf.unstack(ref_batch)):
        pred = list(pred.numpy())
        ref = ref.numpy().decode('utf-8')
        try:
          index = pred.index(0)
          pred = pred[:index][1:-1]
        except:
          pred = pred[1:-1]
        tokenized_string_output = tokenizer.decode([int(p) for p in pred])
    
        file_gold_labels.write(ref + "\n")
        file_outputs.write(tokenized_string_output + "\n")
      
    file_gold_labels.close()
    file_outputs.close()
    score = 100.*bleu_hook.bleu_wrapper(ref_fname, hyp_fname)
    print("Model {}: alpha {}; Beam size {}; Score {:.4f}".format(resultsFileName, alpha, beam_size, score))

class EncoderModel(tf.keras.models.Model):
  def __init__(self,
               input_dim=32000,
               hidden_dim=512,
               projection_type='random',
               num_head=8,
               num_layers_per_block=3,
               num_blocks=2,
               dropout=0.1,
               **kwargs):
    super(EncoderModel, self).__init__()
    self.tiedEmbedding = core_layers.EmbeddingSharedWeights(vocab_size=input_dim, 
                                                hidden_dim=hidden_dim)
    self.pos_encoding = utils.positional_encoding(10000, 
                                            hidden_dim)
    self.encoder = TElayers.Encoder(hidden_dim=hidden_dim,
                           num_head=num_head,
                           projection_type=projection_type,
                           num_blocks=num_layers_per_block,
                           num_encoder=num_blocks,
                           dropout = dropout)
  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, src_inp, training):

    src_maxlen = tf.shape(src_inp)[1]
    enc_mask = utils.create_padding_mask(src_inp)
    enc_in = self.tiedEmbedding(src_inp) + self.pos_encoding[:, :src_maxlen, :]
    v_enc = self.encoder(enc_in, enc_mask, training)

    return v_enc

class ClassificationModel(tf.keras.models.Model):
  def __init__(self,
               num_encoder=1,
               input_dim=32000,
               hidden_dim=512,
               num_head=8,
               projection_type='random',
               num_layers_per_block=3,
               num_blocks=2,
               num_classes=4,
               dropout=0.1,
               **kwargs):
    super(ClassificationModel, self).__init__()
    self.encoder = EncoderModel(input_dim=input_dim,
                          hidden_dim=hidden_dim,
                          num_head=num_head,
                          projection_type=projection_type,
                          num_layers_per_block=num_layers_per_block,
                          num_blocks=num_blocks,
                          dropout = dropout)
    self.out_norm = tf.keras.layers.LayerNormalization()
    
    if num_classes>2:
      self.dense_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    else:
      self.dense_layer = tf.keras.layers.Dense(num_classes-1, activation='sigmoid')
  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, src_inp, training):
    src_maxlen = tf.shape(src_inp)[1]
    v_enc = self.encoder(src_inp, training)
    v = tf.reduce_mean(v_enc, axis=1)
    v = self.out_norm(v)
    d = self.dense_layer(v)        
    return d
