import numpy as np
import pandas as pd
import nltk
from collections import defaultdict



EMBEDDING_DIMENSION = 50
PAD_TOKEN = 0
VOCAB_SIZE = 400001


class dataset(object):
    def __init__(self, s1, s2, tags1, tags2, position1, position2, label):
        self.index_in_epoch = 0
        self.s1 = s1
        self.s2 = s2
        self.tags1 = tags1
        self.tags2 = tags2
        self.position1 = position1
        self.position2 = position2
        self.label = label
        self.num_examples = len(label)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            # print(perm[0:100])
            # print(len(self.s1))
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.tags1 = self.tags1[perm]
            self.tags2 = self.tags2[perm]
            self.position1 = self.position1[perm]
            self.position2 = self.position2[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return np.array(self.s1[start:end]), np.array(self.s2[start:end]), \
               np.array(self.tags1[start:end]), np.array(self.tags2[start:end]), \
               np.array(self.position1[start:end]), np.array(self.position2[start:end]), \
               np.array(self.label[start:end])


def build_dic(filename):
    word2idx = defaultdict(dict)
    weights = []
    f = open(filename, 'r', encoding='utf-8')
    for index, line in enumerate(f):
        values = line.split(' ')
        word = values[0]
        word_weights = values[1:]  # np.asarray(values[1:], dtype=np.float32)
        word2idx[word] = index
        weights.append(word_weights)
        '''if index + 1 == 400000:
            break'''
    print('Loaded Glove!')
    f.close()
    weights = np.asarray(weights)
    return word2idx, weights


def padding_sentence(s1, s2):
    '''
        得到句子s1,s2以后, 映射为id,并用0进行填充，全部填充为102维
        然后用<unk>对句子进行填充
    :param s1:
    :param s2:
    :return:
    '''
    sentence_num = s1.shape[0]
    print(sentence_num)

    s1_padding = np.zeros([sentence_num, 102], dtype=int)
    s2_padding = np.zeros([sentence_num, 102], dtype=int)

    for i, s in enumerate(s1):
        '''if len(s) == 77:
            print(s)'''
        s1_padding[i][:len(s)] = s

    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s

    print('Finished sentence padding!')
    return s1_padding, s2_padding


def get_id(word):
    return word2idx.get(word, 0)
        #return word2idx[word]
    #except:
        #return int('0')


def sent2id(sent):
    sent_split = nltk.word_tokenize(sent.strip())
    features = [get_id(word) for word in sent_split]
    return features


def read_data_sets(filepath):
    df_sick = pd.read_csv(filepath, sep='|',
                          names=['s1', 's2', 'tags1', 'tags2', 'relative_position1',
                                 'relative_position2', 'temp_label'],
                          usecols=[0, 1, 2, 3, 4, 5, 6],
                          dtype={'s1': object, 's2': object, 'tags1': object, 'tags2': object,
                                 'relative_position1': object, 'relative_position2': object,
                                 'temp_label': object},
                          encoding='utf-8'
                          )
    s1 = df_sick.s1.values
    # print(s1)
    s2 = df_sick.s2.values
    tags1 = df_sick.tags1.values
    tags2 = df_sick.tags2.values
    relative_position1 = df_sick.relative_position1.values
    relative_position2 = df_sick.relative_position2.values
    temp_label = df_sick.temp_label.values
    # label = np.asarray(df_sick.label.values, dtype=np.float32)#map(float, df_sick.label.values)
    sample_num = len(temp_label)
    label = []
    m = 0
    for _ in temp_label:
        if _ == '1':
            m = m + 1
            label.append([1, 0])
        else:
            label.append([0, 1])
    label = np.array(label)
    print(m)
    # print(sample_num)
    # print(np.sum(label == '1'))
    # 引入embedding矩阵和字典
    global word2idx, embedding
    word2idx, embedding = build_dic('.\\file\\glove.6B.50d.txt')
    
    # 将word转换为id
    # print(len(s1))
    # print(type(tags1[0]))
    s1 = np.asarray([sent2id(sent) for sent in s1])  # np.array(map(sent2id, s1))
    s2 = np.asarray([sent2id(sent) for sent in s2])  # np.array(map(sent2id, s2))
    # print(s1)
    # 填充句子
    s1_pad, s2_pad = padding_sentence(s1, s2)
    '''new_index = np.random.permutation(sample_num)
    s1_pad = s1_pad[new_index]
    s2_pad = s2_pad[new_index]
    tags1 = tags1[new_index]
    tags2 = tags2[new_index]
    relative_position1 = relative_position1[new_index]
    relative_position2 = relative_position2[new_index]
    label = label[new_index]'''

    return s1_pad, s2_pad, tags1, tags2, relative_position1, relative_position2, label


def read_data_sets1(filepath):
    s1 = []
    s2 = []
    tags1 = []
    tags2 = []
    relative_position1 = []
    relative_position2 = []
    temp_label = []
    label = []
    m =0
    f = open(filepath, 'r', encoding='utf-8')
    for line in f.readlines():
        items = line.strip('\n').split('|')
        s1.append(items[0])
        s2.append(items[1])
        tags1.append(items[2])
        tags2.append(items[3])
        relative_position1.append(items[4])
        relative_position2.append(items[5])
        temp_label.append(items[6])
    sample_num = len(temp_label)
    for _ in temp_label:
        if _ == '1':
            m = m + 1
            label.append([1, 0])
        else:
            label.append([0, 1])
    label = np.array(label)
    tags1 = np.array(tags1)
    tags2 = np.array(tags2)
    relative_position1 = np.array(relative_position1)
    relative_position2 = np.array(relative_position2)
    print(m)
    # print(sample_num)
    # print(np.sum(label == '1'))
    # 引入embedding矩阵和字典
    global word2idx, embedding
    word2idx, embedding = build_dic('.\\file\\glove.6B.50d.txt')

    s1 = np.asarray([sent2id(sent) for sent in s1])  # np.array(map(sent2id, s1))
    s2 = np.asarray([sent2id(sent) for sent in s2])  # np.array(map(sent2id, s2))
    # print(s1)
    # 填充句子
    s1_pad, s2_pad = padding_sentence(s1, s2)
    new_index = np.random.permutation(sample_num)
    s1_pad = s1_pad[new_index]
    s2_pad = s2_pad[new_index]
    tags1 = tags1[new_index]
    tags2 = tags2[new_index]
    relative_position1 = relative_position1[new_index]
    relative_position2 = relative_position2[new_index]
    label = label[new_index]

    return s1_pad, s2_pad, tags1, tags2, relative_position1, relative_position2, label



def batch_iter(data, batch_size, num_epoches, shuffle=True):
    '''
    generate a batch iterator for a dataset
    :param data:
    :param batch_size:
    :param num_epoches:
    :param shuffle:
    :return:
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/data_size) + 1
    for epoch in range(num_epoches):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index, end_index]


# word2idx, embedding = build_dic('G:\glove.6B\glove.6B.50d.txt')
# print(embedding.shape[1])
# nltk.help.upenn_tagset()
# s1_pad, s2_pad, tags1, tags2, relative_position1, relative_position2, label = read_data_sets('G:\\ldc17\\sentpairs1.txt')

'''k, n, t, a, b, c, d = read_data_sets1('.\\file\\sentpairs1.txt')

print(len(k))
print('===========')
print(k[0])'''
'''STS_train = dataset(s1=m, s2=n, label=t)
batch_train = STS_train.next_batch(2)'''
# print(batch_train[0])
# print(type(t))


'''input_s1 = tf.placeholder(dtype=tf.int32, shape=[None, 50], name='input_s1')
input_s2 = tf.placeholder(dtype=tf.int32, shape=[None, 50], name='input_s2')

embedding_size = embedding.shape[1]


# sess.run(W)
W = tf.get_variable('W', shape=embedding.shape, dtype=tf.float32,
                    initializer=tf.constant_initializer(embedding), trainable=True)
s1 = tf.nn.embedding_lookup(W, input_s1)
s2 = tf.nn.embedding_lookup(W, input_s2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(s1, feed_dict={input_s1: m})
    b = sess.run(s2, feed_dict={input_s2: n})
    x = tf.concat([a[0], b[0]], axis=1)
    x = tf.expand_dims(x, -1)
    print(sess.run(x))
    #print(b[0])'''















