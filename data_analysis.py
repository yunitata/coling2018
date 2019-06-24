from __future__ import division, print_function
import lda
import lda.datasets
from sklearn.feature_extraction.text import CountVectorizer
import load_data
from collections import defaultdict
from numpy import sum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from numpy.linalg import norm
from collections import Counter
import argparse
import warnings
np.random.seed(1337)


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ccat10', help='data')
    parser.add_argument('--data_path', type=str, default='/home/yunita/Data/Dataset/Stamatatos/c10_traintest.csv', help='data path')
    parser.add_argument('--n_topics', type=int, default=3, help='number of topics')

    args = parser.parse_args()
    data = args.data
    number_of_topics = args.n_topics
    d_path = args.data_path
    X = []
    dict_author = {}
    if data == "ccat10" or data == 'ccat50':
        X, dict_author = load_data.load_ccat(d_path)
    elif data == "judgment":
        X, dict_author = load_data.load_judgment(d_path)
    elif data == "imdb":
        X, dict_author = load_data.load_imdb62(d_path)

    title = str(data) + " (N_topic = " + str(number_of_topics) + ")"

    # create vocabulary
    print ("creating vocabulary..")
    print ("---------------------------")

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    X_tf = tf_vectorizer.fit_transform(X)
    vocab = tf_vectorizer.get_feature_names()
    print("shape: {}\n".format(X_tf.shape))

    # building topic model using LDA
    print ("building model..")
    print ("---------------------------")
    model = lda.LDA(n_topics=number_of_topics, n_iter=1000, random_state=1000)
    model.fit(X_tf)
    topic_word = model.topic_word_
    print("shape: {}".format(topic_word.shape))

    # show detail of topic
    n = 10
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

    print ("document topic model..")
    print ("---------------------------")
    doc_topic = model.doc_topic_
    topic_most = {}
    for n in range(len(doc_topic)):
        topic_most_pr = doc_topic[n].argmax()
        author = dict_author[n]
        if author in topic_most:
            tp_most.append(topic_most_pr)
        else:
            tp_most = []
            tp_most.append(topic_most_pr)
        topic_most[author] = tp_most

    i = 0
    for author_p, topic_p in topic_most.iteritems():
        print (i, author_p, Counter(topic_p))
        i += 1

    new_dict = defaultdict(list)
    for k, v in dict_author.iteritems():
        new_dict[v].append(k)

    new_dict_2 = defaultdict(list)
    for k, v in new_dict.iteritems():
        sum_per_author = np.zeros(number_of_topics)
        n_doc = len(v)
        for i in range(len(v)):
            sum_per_author = sum([sum_per_author, doc_topic[v[i]]], axis=0)
        mean_prob = sum_per_author/n_doc
        new_dict_2[k].append(mean_prob)

    P1 = []
    j = 0
    for auth, m in new_dict_2.iteritems():
        P1.append(m)
        j +=1

    # calculating JS divergence between authors
    print ("calculating JS divergence..")
    print ("---------------------------")
    P2 = P1
    KL = []
    for p in np.array(P1):
        kl = []
        for q in np.array(P2):
            ent = JSD(p.ravel(), q.ravel())
            kl.append(ent)
        KL.append(kl)
    print ("Average JS Divergence", np.mean(KL))

    # create confusion matrix
    print ("creating heatmap..")
    print ("---------------------------")

    ax = sns.heatmap(np.array(KL))
    ax.set_ylabel('author')
    ax.set_xlabel('author')
    ax.set_title(title)
    ax.collections[0].colorbar.set_label("JS Divergence")
    plt.savefig('heatmap.png')

if __name__ == '__main__':
    main()