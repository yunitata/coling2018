from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import map_tag
from collections import Counter
import nltk
import os
from sklearn.preprocessing import StandardScaler
np.random.seed(1337)


def scaler_(x_train, x):
    # imputer_ = Imputer()
    x_train_transform = np.nan_to_num(x_train)
    x_transform = np.nan_to_num(x)
    scaler = StandardScaler()
    scaler.fit(x_train_transform)
    return np.array(scaler.transform(x_transform))


def tokenize(text):  # tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens

# --- Lexical Features --- #

# Lexical feature (word level)
def average_total_words(x_train):
    sum_wrd = 0
    for text_doc in x_train:
        total_wrd = len(word_tokenize(text_doc))
        sum_wrd += total_wrd
    return sum_wrd/len(x_train)

def total_words(text_doc, x_train):
    avg = average_total_words(x_train)
    total_wrd = len(word_tokenize(text_doc))
    return total_wrd/avg

def average_word_length(text_doc):
    word_list = text_doc.decode('utf-8').split(" ")
    average = sum(len(word) for word in word_list)/len(word_list)
    return average

def total_short_words(text_doc):
    word_list = text_doc.decode('utf-8').split(" ")
    count_short_word = 0
    for word in word_list:
        if len(word) < 4:
            count_short_word += 1
    return count_short_word/len(word_list)

# Lexical feature (character level)

def avg_total_char(x_train):
    sum_chr = 0
    for text_doc in x_train:
        total_chr = len(text_doc)
        sum_chr += total_chr
    return sum_chr/len(x_train)

def total_char(text_doc, x_train):
    avg = avg_total_char(x_train)
    return len(text_doc)/avg

def total_digit(text_doc):
    return sum(c.isdigit() for c in text_doc)/len(text_doc)

def total_uppercase(text_doc):
    return sum(1 for c in text_doc if c.isupper())/len(text_doc)


# Lexical feature (letter frequency)

def count_letter_freq(text_doc):  # per document (vector with length 26)
    text_doc = ''.join([i for i in text_doc if i.isalpha()])
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's'
              , 't', 'u', 'v', 'w', 'x', 'y', 'z']
    count = {}
    for s in text_doc:
      if count.has_key(s):
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in letter:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0
    return np.array(count_list.values())/len(text_doc)
# Lexical feature (digit frequency)


def count_digit_freq(text_doc):   # per document (vector with length 10)
    text_doc = ''.join([i for i in text_doc if i.isdigit()])
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    count = {}
    for s in text_doc:
      if count.has_key(s):
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in digits:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0
    return np.array(count_list.values())/len(text_doc)


def average_sentence_length(text_doc):
    sent_list = sent_tokenize(text_doc.decode('utf-8'), language='english')
    average = sum(len(sent) for sent in sent_list)/len(sent_list)
    return average


# Lexical Feature (vocabulary richness)
def hapax_legomena_ratio(text):  # # per document only a float value
    word_list = text.decode('utf-8').split(" ")
    fdist = nltk.FreqDist(word for word in word_list)
    fdist_hapax = nltk.FreqDist.hapaxes(fdist)
    return float(len(fdist_hapax)/len(word_list))


def dislegomena_ratio(text):  # per document only a float value
    word_list = text.decode('utf-8').split(" ")
    vocabulary_size = len(set(word_list))
    freqs = Counter(nltk.probability.FreqDist(word_list).values())
    VN = lambda i:freqs[i]
    return float(VN(2)*1./vocabulary_size)


def freq_function_word(text):  # per document (vector with length 174)
    words = text.decode('utf-8').split(" ")
    fn = os.path.join(os.path.dirname(__file__), "stopwords.txt")
    with open(fn) as f:
        function_word = f.readlines()
    for i in range(len(function_word)):
        function_word[i] = function_word[i].strip('\n')
    count = {}
    for s in words:
      if count.has_key(s):
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in function_word:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0
    vec = np.array(count_list.values())
    return vec/len(words)


def pos_tag_freq(text):
    pos_tag = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    word_list = text.decode('utf-8').split(" ")
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag)in tag_word)
    count_tag ={}
    for tag in pos_tag:
        freq = tag_fd.get(tag)
        if freq is None:
            count_tag[tag] = 0
        else:
            count_tag[tag] = freq
    return np.array(count_tag.values())/len(word_list)


def punctuation_freq(text):
    punct = ['\'', ':', ',', '_', '!', '?', ';', ".", '\"', '(', ')', '-']
    count = {}
    for s in text:
      if count.has_key(s):
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in punct:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0
    return np.array(count_list.values())/len(text)

def char_bigrams(text, x_train):
    vec = CountVectorizer(analyzer="char", ngram_range=(2, 2), max_df=0.95, min_df=2, max_features=100)
    vec.fit_transform(x_train)
    vocab = vec.vocabulary_
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), vocabulary=vocab, max_features=100)
    vectorizer.fit_transform(x_train)
    feature_bigrams = vectorizer.transform(text)
    return feature_bigrams.toarray()

def char_trigrams(text, x_train):
    vec = CountVectorizer(analyzer="char", ngram_range=(3, 3), max_df=0.95, min_df=2, max_features=100)
    vec.fit_transform(x_train)
    vocab = vec.vocabulary_
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), vocabulary=vocab, max_features=100)
    vectorizer.fit_transform(x_train)
    feature_trigrams = vectorizer.transform(text)
    return feature_trigrams.toarray()

def word_unigram(text, x_train):
    vec = CountVectorizer(analyzer="word", ngram_range=(1, 1), max_df=0.95, min_df=2, max_features=100, stop_words="english")
    vec.fit_transform(x_train)
    vocab = vec.vocabulary_
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1), vocabulary=vocab, max_features=100)
    vectorizer.fit_transform(x_train)
    feature_wunigrams = vectorizer.transform(text)
    return feature_wunigrams.toarray()

def word_bigram(text, x_train):
    vec = CountVectorizer(analyzer="word", ngram_range=(2, 2), max_df=0.95, min_df=2, max_features=100, stop_words="english")
    vec.fit_transform(x_train)
    vocab = vec.vocabulary_
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(2, 2), vocabulary=vocab, max_features=100)
    vectorizer.fit_transform(x_train)
    feature_wbigrams = vectorizer.transform(text)
    return feature_wbigrams.toarray()

def word_trigram(text, x_train):
    vec = CountVectorizer(analyzer="word", ngram_range=(3, 3), max_df=0.95, min_df=2, max_features=100, stop_words="english")
    vec.fit_transform(x_train)
    vocab = vec.vocabulary_
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(3, 3), vocabulary=vocab, max_features=100)
    vectorizer.fit_transform(x_train)
    feature_wtrigrams = vectorizer.transform(text)
    return feature_wtrigrams.toarray()


def create_feature(text_x, x_train, mode):
    # mode 0 : all features are included
    # mode 1 : remove lexical features
    # mode 2 : remove syntactic features
    # mode 3 : remove character features
    # mode 4 : remove content features
    stylometry_vector_all = []
    if mode == 0:
        c_bigram = char_bigrams(text_x, x_train)
        c_trigram = char_trigrams(text_x, x_train)
        w_unigram = word_unigram(text_x, x_train)
        w_bigram = word_bigram(text_x, x_train)
        w_trigram = word_trigram(text_x, x_train)

        for x in text_x:
            stylometry_vector = []
            stylometry_vector.append(average_word_length(x))
            stylometry_vector.append(total_short_words(x))
            stylometry_vector.append(total_digit(x))
            stylometry_vector.append(total_uppercase(x))
            stylometry_vector.extend(count_letter_freq(x))
            stylometry_vector.extend(count_digit_freq(x))
            stylometry_vector.append(hapax_legomena_ratio(x))
            stylometry_vector.append(dislegomena_ratio(x))
            stylometry_vector.extend(freq_function_word(x))
            stylometry_vector.extend(punctuation_freq(x))
            stylometry_vector_all.append(stylometry_vector)
        b = np.array(stylometry_vector_all)
        a = np.concatenate((c_bigram, c_trigram, w_unigram, w_bigram, w_trigram, b), axis=1)
        return a

    elif mode == 1:
        c_bigram = char_bigrams(text_x, x_train)
        c_trigram = char_trigrams(text_x, x_train)
        w_unigram = word_unigram(text_x, x_train)
        w_bigram = word_bigram(text_x, x_train)
        w_trigram = word_trigram(text_x, x_train)
        a = np.concatenate((c_bigram, c_trigram, w_unigram, w_bigram, w_trigram), axis=1)
        return a
    elif mode == 2:
        c_bigram = char_bigrams(text_x, x_train)
        c_trigram = char_trigrams(text_x, x_train)
        for x in text_x:
            stylometry_vector = []
            stylometry_vector.append(average_word_length(x))
            stylometry_vector.append(total_short_words(x))
            stylometry_vector.append(total_digit(x))
            stylometry_vector.append(total_uppercase(x))
            stylometry_vector.extend(count_letter_freq(x))
            stylometry_vector.extend(count_digit_freq(x))
            stylometry_vector.append(hapax_legomena_ratio(x))
            stylometry_vector.append(dislegomena_ratio(x))
            stylometry_vector.extend(freq_function_word(x))
            stylometry_vector.extend(punctuation_freq(x))
            stylometry_vector_all.append(stylometry_vector)
        b = np.array(stylometry_vector_all)
        a = np.concatenate((c_bigram, c_trigram, b), axis=1)
        return a
    elif mode == 3:
        w_unigram = word_unigram(text_x, x_train)
        w_bigram = word_bigram(text_x, x_train)
        w_trigram = word_trigram(text_x, x_train)
        for x in text_x:
            stylometry_vector = []
            stylometry_vector.append(average_word_length(x))
            stylometry_vector.append(total_short_words(x))
            stylometry_vector.append(total_digit(x))
            stylometry_vector.append(total_uppercase(x))
            stylometry_vector.extend(count_letter_freq(x))
            stylometry_vector.extend(count_digit_freq(x))
            stylometry_vector.append(hapax_legomena_ratio(x))
            stylometry_vector.append(dislegomena_ratio(x))
            stylometry_vector.extend(freq_function_word(x))
            stylometry_vector.extend(punctuation_freq(x))
            stylometry_vector_all.append(stylometry_vector)
        b = np.array(stylometry_vector_all)
        a = np.concatenate((w_unigram, w_bigram, w_trigram, b), axis=1)
        return a


def feature_extraction(x_train, x_val, x_test, feature_code):
    print ("create features for training set....")
    x_train_features = create_feature(x_train, x_train, feature_code)
    print ("create feature for validation set")
    x_val_features = create_feature(x_val, x_train,feature_code)
    print ("create feature for testing set")
    x_test_features = create_feature(x_test, x_train, feature_code)
    return np.array(x_train_features), np.array(x_val_features), np.array(x_test_features)
