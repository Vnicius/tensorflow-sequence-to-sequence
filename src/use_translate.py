import sys

import tensorflow as tf
import dill as pickle
from nltk import word_tokenize

from Seq2seq import Seq2seq
import helpers

def do_prediction(sess, sequence):
      encoder_inputs_, _ = helpers.batch(sequence)
      decoder_inputs_, _ = helpers.batch(
          [[0] + (sequence) for sequence in sequence]
      )

      #return(sess.run(self.prediction, feed_dict={
      return(sess.run(tf.get_collection('prediction'), feed_dict={       
        tf.get_default_graph().get_tensor_by_name("encoder_inputs:0"): encoder_inputs_,
        tf.get_default_graph().get_tensor_by_name("decoder_inputs:0"): decoder_inputs_
      }))

if __name__ == "__main__":
    try:
        print("Carregando dicionários...")
        pickle_src = open(".pickle/" + sys.argv[1] + ".pkl", "rb")
        src_dict = pickle.load(pickle_src)
        pickle_src.close()

        pickle_tgt = open(".pickle/" + sys.argv[2] + ".pkl", "rb")
        tgt_dict = pickle.load(pickle_tgt)
        pickle_tgt.close()

    except:
        sys.exit("Dicionários não encontrados!")
    
    text = sys.argv[3]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('.model/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('.model/'))
        sess.run(tf.global_variables_initializer())

        res = do_prediction(sess, [list(src_dict.sentence2num(text))])

        print(list(tgt_dict.num2sentence([i[0] for i in res[0]])))


