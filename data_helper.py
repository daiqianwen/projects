import os
from collections import defaultdict
from nltk import word_tokenize


def eachFile(filepath):
    filenames = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s//%s' % (filepath, allDir))
        filenames.append(child)
    return filenames


def get_ems(file_path):
    eachfileems = []
    f = open(file_path, 'r', encoding='utf-8')

    for line in f.readlines():
        eventmention = defaultdict(dict)
        items = line.split('|')
        # print(items)
        eventmention['source'] = items[0]
        eventmention['hopper_id'] = items[1]
        eventmention['em_id'] = items[2]
        eventmention['type'] = items[3]
        eventmention['subtype'] = items[4]
        eventmention['realis'] = items[5]
        eventmention['trigger'] = items[6]
        eventmention['position'] = int(items[7])
        eventmention['sent_index'] = int(items[8])
        eventmention['sentence'] = items[9]
        eventmention['tag'] = items[10]
        # print(items[11])
        tags = items[11].split(' ')
        eventmention['tags'] = tags
        # print(items[12])
        argroles = items[12].split('#')
        n = len(argroles)
        argroles = argroles[0:n-1]
        arg_role = []

        for argrole in argroles:
            ar = argrole.split('+')
            # print(ar)
            arg_role.append((ar[0], ar[1]))
        eventmention['arg_role'] = arg_role
        # print(eventmention)
        eachfileems.append(eventmention)
    f.close()
    return eachfileems


def get_sent_pair(ems):
    n = len(ems)
    pairs = []
    labels = []
    pos = 0
    neg = 0
    for i in range(n):
        flag = 0
        for j in range(i+1, n):
            if ems[i]['type'] == ems[j]['type']:
                label = 1
                pair = defaultdict(dict)
                pair['sentence1'] = ems[i]['sentence']
                pair['sentence2'] = ems[j]['sentence']
                pair['tags1'] = ems[i]['tags']
                pair['tags2'] = ems[j]['tags']
                pair['relative_position1'] = \
                    [(k - ems[i]['position']) for k in range(len(word_tokenize(pair['sentence1'])))]
                pair['relative_position2'] = \
                    [(k - ems[j]['position']) for k in range(len(word_tokenize(pair['sentence2'])))]
                sent_pair = ems[i]['sentence'] + '|' + ems[j]['sentence']
                print(sent_pair)
                # print(pair)

                if ems[i]['hopper_id'] == ems[j]['hopper_id']:
                    label = 1
                    pos += 1
                    pairs.append(pair)
                    labels.append(label)
                else:
                    label = -1
                    neg += 1
                    if i >= 7:
                        continue
                    pairs.append(pair)
                    labels.append(label)

    return pairs, labels


def sent_pairs2txt(into_path):
    m = 0
    n = 0
    f = open(into_path, 'w', encoding='utf-8')
    filenames = eachFile('G:\\ldc17\\fileems1')
    for filename in filenames:
        ems = get_ems(filename)
        '''for em in ems:
            print(em)'''
        pairs, labels = get_sent_pair(ems)
        for i, pair in enumerate(pairs):
            f.write(pair['sentence1'] + '|' +
                    pair['sentence2'] + '|')
            for tag in pair['tags1']:
                f.write(tag+' ')
            f.write('|')
            for tag in pair['tags2']:
                f.write(tag + ' ')
            f.write('|')
            for re_position in pair['relative_position1']:
                f.write(str(re_position) + ' ')
            f.write('|')
            for re_position in pair['relative_position2']:
                f.write(str(re_position) + ' ')
            f.write('|')
            f.write(str(labels[i]) + '\n')
            if labels[i] == 1:
                m = m+1
            else:
                n = n+1
            #  f.write(pair.strip().lower() + ' ' + str(label) + '\n')
    f.close()
    print(m)
    print(n)

sent_pairs2txt('G:\\ldc17\\sentpairs1.txt')
# get_ems('G:\ldc17\\fileems\\ENG_DF_000170_20131217_G00A0BFNB.txt')