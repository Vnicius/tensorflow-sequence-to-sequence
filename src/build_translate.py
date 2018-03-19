import tensorflow as tf
from Seq2seq import Seq2seq
from dictionary import Dictionary
from nltk import word_tokenize

# def get_batch(batch_size, batch_src_it, batch_tgt_it):
#     return [list(next(batch_src_it)) for _ in range(batch_size)],
#         [list(next(batch_tgt_it)) for _ in range(batch_size)]

if __name__ == "__main__":
    
    # src_file_path = './europarl-v7.pt-en.en'
    # tgt_file_path = './europarl-v7.pt-en.pt'

    # src_file_path = './europarl-v7.pt-en.en.trunc'
    # tgt_file_path = './europarl-v7.pt-en.pt.trunc'

    src_file_path = './ingles.txt'
    tgt_file_path = './asl.txt'

    #data_src, data_tgt = gen(100100)
    print("Criando dicionário da origem")
    src_dict = Dictionary(src_file_path, lambda text: word_tokenize(text))
    src_dict.build_dictionary()

    print("Criando dicionário do objetivo")
    tgt_dict = Dictionary(tgt_file_path, lambda text: word_tokenize(text, 'portuguese'))
    tgt_dict.build_dictionary()

    batch_src_it = None
    batch_tgt_it = None

    print("Criando batches da origem")
    # with open(src_file_path, 'r') as src_file:
    #     for line in src_file.readlines()[:50000]:
    #         line = line.replace('\n', '')
    #         batch_src.append(src_dict.sentence2num(line))

    batch_src_it = (src_dict.sentence2num(line.replace('\n', ''))
                    for line in open(src_file_path, 'r'))

    print("Criando bathches do objetivo")
    # with open(tgt_file_path, 'r') as tgt_file:
    #     for line in tgt_file.readlines()[:50000]:
    #         line = line.replace('\n', '')
    #         batch_tgt.append(tgt_dict.sentence2num(line))

    
    batch_tgt_it = (tgt_dict.sentence2num(line.replace('\n', ''))
                    for line in open(tgt_file_path, 'r'))
    
    #batches = get_batch(2, batch_src_it, batch_tgt_it)

    #x, y = next(batches)

    batch_size = 4

    def next_batches():
        src = [list(next(batch_src_it)) for _ in range(batch_size)]
        tgt = [list(next(batch_tgt_it)) for _ in range(batch_size)]

        return src,tgt
    
    seq2seq = Seq2seq(len(src_dict.word2num_dict),len(tgt_dict.word2num_dict),
                      32, 256, 256)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    seq2seq.train(next_batches, 60)
        #print(src_dict.num2sentence(batch_src[4000]))
        ###########
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = list(next(batch_src_it))
        print(list(src_dict.num2sentence(test)))
        print()
        res = seq2seq.do_prediction(sess, [test])

        print(list(tgt_dict.num2sentence([i[0] for i in res])))