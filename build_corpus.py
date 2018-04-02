import re
import os
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from xml.etree import ElementTree
from collections import defaultdict


def eachFile(filepath):
    filenames=[]
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s//%s' % (filepath, allDir))
        filenames.append(child)
    return filenames


def sentence_split(str_sentence):
    """
    分句
    Args:
        str_sentence: str, 文本内容
    Returns:
         list_ret: list, 完成分句的句子列表
    """
    clean_con = re.sub('<[^>]*>', '', str_sentence)
    # clean_con = clean_con.replace('\n','')
    list_ret = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for s_str in clean_con.split('\n'):
        s = tokenizer.tokenize(s_str)
        list_ret = list_ret + s
    # print(list_ret)
    return list_ret


def get_extend_attrib(source_path, offset, text):
    """
    获取事件所在的句子、词性、相对位置等信息
    Args:
        source_path: str， 原文路径
        offset: int， 触发词在原文中的位置
        text: str, 触发词
    returns:
        sentence: str,  句子
        tags: str, 词性标签
        position: int, 相对位置
    """
    context = open(source_path, 'r', encoding='utf-8').read()
    tempcon = context[0:offset]
    sent_index = len(sentence_split(tempcon)) - 1
    startoff, endoff = 0, 0  # 句子的始末位置
    for index in range(offset, -1, -1):  # 向前查找
        if context[index] in ['"', '.', '!', '。', '?', '\n', '>']:
            startoff = index + 1
            break
    for index in range(offset, len(context), 1):  # 向后查找
        if context[index] in ['.', '!', '。', '?', '\n', '<']:
            endoff = index + 1
            break
    sentence = context[startoff: endoff]
    sentence = sentence.strip('\n')
    sentence = sentence.lower().replace('↑', ':')
    words = word_tokenize(sentence)  # 分词
    position = 0
    try:
        position = words.index(word_tokenize(text)[0])   # 触发词在句中的位置, 对触发词分词的目的防止触发词由多个单词构成, 获取第一个词位置
    except Exception:
        text = word_tokenize(text)[0]
        for index in range(len(words)):
            if text in words[index]:
                position = index
    tagged = pos_tag(words)
    tags = list(map(lambda x: x[1], tagged))
    tag = tags[position]
    tags = ' '.join(tags)  # 词性标注

    return sentence, tags, position, sent_index, tag


def get_eminfo(ere_path, source_path):
    eachfileems = []
    tree = ElementTree.parse(ere_path)
    root = tree.getroot()

    for hoppernode in root.iter('hopper'):
        hopper1 = hoppernode.get('id')
        for eventnode in hoppernode.iter('event_mention'):
            eventmention = defaultdict(dict)
            eventmention['hopper_id'] = hopper1
            arg_role = []
            id1 = eventnode.get('id')
            eventmention['em_id'] = id1

            type1 = eventnode.get('type')
            eventmention['type'] = type1

            subtype1 = eventnode.get('subtype')
            eventmention['subtype'] = subtype1

            realis1 = eventnode.get('realis')
            eventmention['realis'] = realis1

            triggernode = eventnode.find('trigger')
            source1 = triggernode.get('source')
            eventmention['source'] = source1

            offset1 = triggernode.get('offset')
            trigger1 = triggernode.text

            lem = WordNetLemmatizer()
            trigger_etyma1 = lem.lemmatize(trigger1, "v")
            eventmention['trigger'] = trigger_etyma1

            sentence, tags, position, sent_index, tag = get_extend_attrib(source_path, int(offset1), trigger1)
            #print(tags.split(' '))
            eventmention['sentence'] = sentence
            eventmention['tags'] = tags
            eventmention['position'] = position
            eventmention['sent_index'] = sent_index
            eventmention['tag'] = tag

            for em_arg in eventnode.iter('em_arg'):
                arg1 = em_arg.text
                role1 = em_arg.get('role')
                arg_role.append((arg1, role1))
            eventmention['arg_role'] = arg_role
            eachfileems.append(eventmention)

    return eachfileems


def ems2txt(eachfileems, txtpath):
    f = open(txtpath, 'w', encoding='utf-8')
    for em in eachfileems:
        f.write(em['source'] + '|' + em['hopper_id'] + '|' + em['em_id'] + '|' + em['type'] + '|' +
                em['subtype'] + '|' + em['realis'] + '|' + em['trigger'] + '|' + str(em['position']) + '|'
                + str(em['sent_index']) + '|' + em['sentence'] + '|' + em['tag'] + '|' + em['tags'] + '|')
        '''for item in em['tags']:
            f.write(item + ' ')
        f.write('|')'''
        for item in em['arg_role']:
            f.write(item[0] + '+' + item[1] + '#')
        f.write('\n')
    f.close()


ere_paths = eachFile('G:\ldc17\eresour\ere')
source_paths = eachFile('G:\ldc17\eresour\source')
n = len(ere_paths)
for i in range(n):
    ere_path = ere_paths[i]
    source_path = source_paths[i]
    name = os.path.basename(source_path)
    name = name[:-4] + '.txt'
    into_path = "G:\\ldc17\\fileems1" + '\\' + name
    #print(into_path)
    evs = get_eminfo(ere_path, source_path)
    print(evs)
    ems2txt(evs, into_path)
