import pandas as pd
import csv, re, sys
import codecs
import numpy as np
from sklearn import preprocessing

csv.field_size_limit(sys.maxsize)
np.random.seed(1337)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\d+", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_ccat(data_path):
    # load data (can be used if data already split into train and test set)
    data = pd.read_csv(data_path, header=0)
    x = data['article']
    y = data['class']
    dict_author = {}    # id doc: author_name
    X = []
    for i in range(len(x)):
        X.append(clean_str(x[i]))
        dict_author[i] = y[i]
    auth_class = list(set(y))
    le = preprocessing.LabelEncoder()
    le.fit(auth_class)
    y_numeric = le.transform(y)
    return X, dict_author, np.array(x), np.array(y_numeric)


def load_ccat_data(data_path):
    # load data (can be used if data already split into train and test set)
    data = pd.read_csv(data_path, header=0)
    x = np.array(data['article'])
    y = np.array(data['class'])
    #x_mapped = text_preprocess.mapped_text(x)  # mapping with listed character
    # transform y (label of author name) into integer label (start from 1)
    auth_class = list(set(y))
    le = preprocessing.LabelEncoder()
    le.fit(auth_class)
    print le
    y_numeric = le.transform(y)
    return np.array(x), np.array(y_numeric)



def load_judgment(data_judgment):
    with codecs.open(data_judgment, 'rb') as f:
        reading = csv.reader(f, delimiter='\t')
        author = []
        content = []
        for row in reading:
            auth = row[1].split(".")
            auth_class = re.sub(r"[^A-Za-z]+", '', auth[0])
            if row[0].lower() == "rich1913-1928" or row[0].lower() == "dixon" or row[0].lower() == "mctiernan1965-1975":
                author.append(auth_class.lower())
                content.append(row[2].decode('utf-8'))
        x = content
        y = author

    dict_author = {}    # id doc: author_name
    X = []
    for i in range(len(x)):
        X.append(clean_str(x[i]))
        dict_author[i] = y[i]
    return X, dict_author


def load_imdb62(data_path_imdb):
    with open(data_path_imdb, 'rb') as f:
        reading = csv.reader(f, delimiter='\t')
        reading.next()
        author = []
        content = []
        for row in reading:
            author.append(row[0])
            content.append(row[1].decode('utf-8'))
        x = content
        y = author
    x_ = []
    y_ = []
    author_list = list(set(y))
    for auth in author_list:
        i = 0
        for p in range(len(y)):
            if auth == y[p]:
                i += 1
                x_.append(x[p])
                y_.append(y[p])
    dict_author = {}    # id doc: author_name
    X = []
    for i in range(len(x_)):
        X.append(clean_str(x_[i]))
        dict_author[i] = y_[i]
    return X, dict_author