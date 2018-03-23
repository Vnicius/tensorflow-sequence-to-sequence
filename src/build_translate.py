import sys
from random import shuffle
import time

import tensorflow as tf
from Seq2seq import Seq2seq
from dictionary import Dictionary
from nltk import word_tokenize
import dill as pickle

# def get_batch(batch_size, batch_src_it, batch_tgt_it):
#     return [list(next(batch_src_it)) for _ in range(batch_size)],
#         [list(next(batch_tgt_it)) for _ in range(batch_size)]

class ReadFileIterator:
    def __init__(self, file_path, start=0, end=0):
        self.lines = open(file_path, 'r').readlines()
        self.start = 0 if not start else int(start * len(self.lines))
        self.end = len(self.lines) if not end else int(end * len(self.lines))

    def __inter__(self):
        return self
    
    def __next__(self):
        index = 0
        
        if self.start < self.end:
            index = self.start
            self.start += 1
        else:
            self.start = 0 if not start else int(start * len(self.lines))

        return self.lines[index]

if __name__ == "__main__":
    
    src_file_path = './.data/europarl-v7.pt-en.en'
    tgt_file_path = './.data/europarl-v7.pt-en.pt'

    # src_file_path = './europarl-v7.pt-en.en.trunc'
    # tgt_file_path = './europarl-v7.pt-en.pt.trunc'

    # src_file_path = './ingles.txt'
    # tgt_file_path = './asl.txt'

    #data_src, data_tgt = gen(100100)

    try:
        print("Carregando dicionários...")
        pickle_src = open(".pickle/" + sys.argv[1] + ".pkl", "rb")
        src_dict = pickle.load(pickle_src)
        pickle_src.close()

        pickle_tgt = open(".pickle/" + sys.argv[2] + ".pkl", "rb")
        tgt_dict = pickle.load(pickle_tgt)
        pickle_tgt.close()

    except:
        print("Erro...")
        print("Criando dicionário da origem")
        src_dict = Dictionary(src_file_path, lambda text: word_tokenize(text))
        src_dict.build_dictionary()
        pickle.dump(src_dict, open(".pickle/" + sys.argv[1] + ".pkl", "wb"))

        print("Criando dicionário do objetivo")
        tgt_dict = Dictionary(tgt_file_path, lambda text: word_tokenize(text, 'portuguese'))
        tgt_dict.build_dictionary()
        pickle.dump(tgt_dict, open(".pickle/" + sys.argv[2] + ".pkl", "wb"))

    batch_src_it = None
    batch_tgt_it = None

    src_it = ReadFileIterator(src_file_path, end=0.8)
    tgt_it = ReadFileIterator(tgt_file_path, end=0.8)

    valid_src_it = ReadFileIterator(src_file_path, start=0.8, end=0.9)
    valid_tgt_it = ReadFileIterator(tgt_file_path, start=0.8, end=0.9)

    print("Criando batches da origem")
    # with open(src_file_path, 'r') as src_file:
    #     for line in src_file.readlines()[:50000]:
    #         line = line.replace('\n', '')
    #         batch_src.append(src_dict.sentence2num(line))

    #batch_src_it = lambda  : src_dict.sentence2num(next(src_it).replace('\n', ''))

    batch_it = lambda it : src_dict.sentence2num(next(it).replace('\n', ''))

    print("Criando bathches do objetivo")
    # with open(tgt_file_path, 'r') as tgt_file:
    #     for line in tgt_file.readlines()[:50000]:
    #         line = line.replace('\n', '')
    #         batch_tgt.append(tgt_dict.sentence2num(line))


    #batch_tgt_it = lambda : tgt_dict.sentence2num(next(tgt_it).replace('\n', ''))
    
    #batches = get_batch(2, batch_src_it, batch_tgt_it)

    #x, y = next(batches)

    batch_size = 25

    # def resetIt(src_file, tgt_file):
    #     src_file.close()
    #     tgt_file.close()

    #     src_file = open(src_file_path, 'r')
    #     tgt_file = open(tgt_file_path, 'r')

    #     batch_src_it = (src_dict.sentence2num(line.replace('\n', ''))
    #                 for line in src_file)
        
    #     batch_tgt_it = (tgt_dict.sentence2num(line.replace('\n', ''))
    #                 for line in tgt_file)

    def next_batches():
        batches = [[list(batch_it(src_it)), list(batch_it(tgt_it))] for _ in range(batch_size)]

        shuffle(batches)
        src = [batch[0] for batch in batches]
        tgt = [batch[1] for batch in batches]

        return src,tgt
    
    def validation_batches():
        src = [list(batch_it(valid_src_it)) for _ in range(batch_size)]
        tgt = [list(batch_it(valid_tgt_it)) for _ in range(batch_size)]

        return src, tgt
    
    seq2seq = Seq2seq(len(src_dict.word2num_dict),len(tgt_dict.word2num_dict),
                      32, 256, 256)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    start_train = time.strftime("%d/%m/%Y %H:%M:%S")
    print("\033[94m"+"START TRAIN:", start_train, '\033[0m')

    seq2seq.train(next_batches, validation_batches, 100,
                  int(src_it.end - src_it.start/batch_size),
                  int(valid_src_it.end - valid_src_it.start))
        #print(src_dict.num2sentence(batch_src[4000]))
        ###########
    print("\033[92m"+"START TRAIN:", start_train, '\033[0m')
    print("\033[92m"+"END TRAIN:", time.strftime("%d/%m/%Y %H:%M:%S"), '\033[0m')
    
    #pickle.dump(seq2seq, open('.pickle/seq2seq.pkl','wb'))

    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     try:
    #       saver.restore(sess, '.model/model.ckpt')
    #       print('Restored...')
    #     except:
    #       pass

    #     sess.run(tf.global_variables_initializer())
    #     test = list(batch_src_it())
 
    #     print(list(src_dict.num2sentence(test)))
    #     print("\nEsperado:", next(tgt_it))
    #     print()
    #     res = seq2seq.do_prediction(sess, [test])

    #     print(list(tgt_dict.num2sentence([i[0] for i in res[0]])))
        