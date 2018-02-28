
from collections import Counter

class Dictionary():
    def __init__(self, src_file, tokenizer=None):
        self._src_file = src_file
        self.tokenizer = tokenizer
        self.num2word_dict = {0 : 'PAD', 1 : 'EOS', 2 : 'UKN'} # dicionário com os indices e suas respectivas palavras
        self.word2num_dict = {'PAD': 0, 'EOS' : 1, 'UKN' : 2} # dicionário com as palavras e seus respectivos índices

    def build_dictionary(self):
        with open(self._src_file, 'r') as fl:
            it = (line.replace('\n', '') for line in fl)
            i = 0
            while i <= 10000:
                try:
                    line = next(it)
                    words_list = []

                    if self.tokenizer:
                        words_list = self.tokenizer(line)
                    else:
                        words_list = line.split(' ')

                    for word in words_list:
                        try:
                            self.word2num_dict[word]
                        except KeyError:
                            self.word2num_dict[word] = len(self.word2num_dict)
                            self.num2word_dict[len(self.num2word_dict)] = word
                    
                    i += 1
                except StopIteration:
                    break
    
    def sentence2num(self, sentence):
        # words_list = []
        # num_list = []

        # if self.tokenizer:
        #     words_list = self.tokenizer(sentence)
        # else:
        #     words_list = sentence.split(' ')
        
        # for word in words_list:
        #     try:
        #         num_list.append(self.word2num_dict[word])
        #     except KeyError:
        #         num_list.append(self.word2num_dict['UKN'])

        # return num_list

        words_list = []

        if self.tokenizer:
            words_list = self.tokenizer(sentence)
        else:
            words_list = sentence.split(' ')
        
        for word in words_list:
            try:
                yield self.word2num_dict[word]
            except KeyError:
                yield self.word2num_dict['UKN']




    def num2sentence(self, numbers):
        #words_list = []

        # for number in numbers:
        #     try:
        #         words_list.append(self.num2word_dict[number])
        #     except KeyError:
        #         words_list.append('UKN')
        
        # return " ".join(words_list)

        for number in numbers:
            try:
                yield self.num2word_dict[number]
            except KeyError:
                yield 'UKN'
        

if __name__ == '__main__':
    dic = Dictionary('./europarl-v7.pt-en.en')
    dic.build_dictionary()

    arr = dic.sentence2num('I am he')
    print(list(arr))
    print(list(dic.num2sentence([7, 511, 360])))
