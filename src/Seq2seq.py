#!/usr/bin/python3
# -*- coding : uft-8

import tensorflow as tf
import helpers

PAD = 0
EOS = 1

class Seq2seq:
    def __init__(self,
                input_vocab_size,
                output_vocab_size,
                inputs_emb_size=10,
                encoder_hidden_units=32,
                decoder_hidden_units=32,
                ):

      self.input_vocab_size = input_vocab_size
      self.output_vocab_size = output_vocab_size
      self.inputs_emb_size = inputs_emb_size
      self.encoder_hidden_units = encoder_hidden_units
      self.decoder_hidden_units = decoder_hidden_units

      self.encoder_inputs = None
      self.decoder_targets = None
      self.decoder_inputs = None
      self.encoder_inputs_emb = None
      self.decoder_inputs_emb = None
      self.encoder_final_state = None
      self.decoder_outputs = None
      self.decoder_final_state = None
      self.decoder_logits = None
      self.prediction = None

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
      
    def _init_embeddings(self):
      embeddings = tf.Variable(tf.random_uniform([self.input_vocab_size, self.inputs_emb_size], -1.0, 1.0),
                                                  dtype=tf.float32)
      
      self.encoder_inputs_emb = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
      self.decoder_inputs_emb = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
    
    def _build_encoder_cell(self):
      return tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)
    
    def _encoder(self):
      encorder_cell = self._build_encoder_cell()

      _, self.encoder_final_state = tf.nn.dynamic_rnn(
        encorder_cell, self.encoder_inputs_emb,
        dtype=tf.float32, time_major=True
      )
    
    def _build_decoder_cell(self):
      return tf.contrib.rnn.LSTMCell(self.decoder_hidden_units)
    
    def _decoder(self):
      decoder_cell = self._build_decoder_cell()

      self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, self.decoder_inputs_emb,
        initial_state=self.encoder_final_state,
        dtype=tf.float32, time_major=True, scope='plain_decoder'
      )

      self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs,
                                                    self.output_vocab_size)
      self.prediction = tf.argmax(self.decoder_logits, 2)
    
    def get_optimizer(self):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(self.decoder_targets, depth=self.output_vocab_size, dtype=tf.float32),
        logits=self.decoder_logits
      )

      loss = tf.reduce_mean(cross_entropy)
      optimizer = tf.train.AdamOptimizer().minimize(loss)

      return loss, optimizer
    
    def build_model(self):
      self._init_placeholders()
      self._init_embeddings()
      self._encoder()
      self._decoder()
    
    def train(self, batches, epochs):
      # batches = batches = helpers.random_sequences(length_from=3, length_to=8,
      #                              vocab_lower=2, vocab_upper=10,
      #                              batch_size=batch_size)
      self.build_model()
      loss, train_op = self.get_optimizer()

      # def next_feed(encoder_inputs=self.encoder_inputs,
      #               decoder_inputs=self.decoder_inputs,
      #               decoder_targets=self.decoder_targets):
      #   batch = next(batches)
      #   encoder_inputs_, _ = helpers.batch(batch)
      #   decoder_targets_, _ = helpers.batch(
      #       [(sequence) + [EOS] for sequence in batch]
      #   )
      #   decoder_inputs_, _ = helpers.batch(
      #       [[EOS] + (sequence) for sequence in batch]
      #   )
      #   return {
      #       encoder_inputs: encoder_inputs_,
      #       decoder_inputs: decoder_inputs_,
      #       decoder_targets: decoder_targets_,
      #   }

      def next_feed(encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs,
                    decoder_targets=self.decoder_targets):
        
        src_batch, tgt_batch = batches()
        encoder_inputs_, _ = helpers.batch(src_batch, 50)
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [EOS] for sequence in tgt_batch]
        ,50)
        decoder_inputs_, _ = helpers.batch(
            [[EOS] + (sequence) for sequence in src_batch]
        ,50)
        return {
            encoder_inputs: encoder_inputs_,
            decoder_inputs: decoder_inputs_,
            decoder_targets: decoder_targets_,
        }
      
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # max_batches = 5001
        # batches_in_epoch = 1000
        loss_track = []

        try:
            for epoch in range(epochs):
                fd = next_feed()
                _, l = sess.run([train_op, loss], feed_dict=fd)
                loss_track.append(l)

                print('Época: ' + str(epoch))

                # if batch == 0 or batch % batches_in_epoch == 0:
                #     print('batch {}'.format(batch))
                #     print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                #     fd = next_feed()
                #     predict_ = sess.run(self.prediction, fd)
                #     for i, (inp, pred) in enumerate(zip(fd[self.encoder_inputs].T, predict_.T)):
                #         print('  sample {}:'.format(i + 1))
                #         print('    input     > {}'.format(inp))
                #         print('    predicted > {}'.format(pred))
                #         if i >= 2:
                #             break
                #     print()
        except KeyboardInterrupt:
            print('training interrupted')
      
    def do_prediction(self, sess, sequence):
      encoder_inputs_, _ = helpers.batch(sequence)
      decoder_inputs_, _ = helpers.batch(
          [[EOS] + (sequence) for sequence in sequence]
      )

      return(sess.run(self.prediction, feed_dict={
        self.encoder_inputs: encoder_inputs_,
        self.decoder_inputs: decoder_inputs_
      }))
    
if __name__ == '__main__':
  s2s = Seq2seq(10,10)
  s2s.train(100,3)