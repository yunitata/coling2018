from __future__ import division
import load_data
import numpy as np
import warnings
import argparse
from sklearn.cross_validation import StratifiedKFold, train_test_split
import feature_extractor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.utils.np_utils import to_categorical
import model
np.random.seed(1337)


def LogReg(x_tr, y_tr, x_ts, y_ts):
    print ("training using Logistic Regression...")
    log_classifier = LogisticRegression()
    pipe = Pipeline([('standardscaler',
                 StandardScaler(copy=True, with_mean=True, with_std=True)), ('classification', log_classifier)])
    pipe.fit(np.nan_to_num(x_tr), y_tr)
    print ("predicting using LogReg...")
    y_pred = pipe.predict(np.nan_to_num(x_ts))
    acc = accuracy_score(y_ts, y_pred)
    print acc
    return acc


def FNN(x_train_rep, x_val_rep, x_test_rep, y_tr, y_val, y_ts,
        batch_size, nb_class, nb_epoch, lr, md_path, w_path, hidden_size, input_dim, dropout):
    # turn the class label into categorical
    y_tr = to_categorical(y_tr)
    y_val = to_categorical(y_val)
    y_ts = to_categorical(y_ts)

    loss_val, acc_val, loss_test, acc_test = model.model_dense(x_train_rep, x_val_rep, x_test_rep, y_tr, y_val, y_ts,
                       nb_class, nb_epoch, batch_size, lr, md_path, w_path, hidden_size, input_dim, dropout)
    print loss_val, acc_val, loss_test, acc_test
    return loss_val, acc_val, loss_test, acc_test


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ccat10', help='data')
    parser.add_argument('--nb_class', type=int, default=10, help='number of class')
    parser.add_argument('--nb_epoch', type=int, default=250, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate')
    parser.add_argument('--hidden_size', type=int, default=500, help='hidden size')
    parser.add_argument('--model_path', type=str, default="model.h5", help='model path')
    parser.add_argument('--weight_path', type=str, default="weight.json", help='weight path')
    parser.add_argument('--feature_dim', type=int, default=728, help='feature dimension')
    parser.add_argument('--train_path', type=str, default='/home/yunita/Data/Dataset/Stamatatos/c10_train.csv', help='train path')
    parser.add_argument('--test_path', type=str, default='/home/yunita/Data/Dataset/Stamatatos/c10_test.csv', help='test path')
    parser.add_argument('--feature_code', type=int, default=0, help='feature_code')
    # feature code
    # 0 : all features are included
    # 1 : remove style features
    # 2 : remove content/word features
    # 3 : remove hybrid/char features

    args = parser.parse_args()
    data = args.data
    train_path = args.train_path
    test_path = args.test_path
    feature_code = args.feature_code

    if data == 'ccat10' or data == 'ccat50':
        x_train, y_train = load_data.load_ccat_data(train_path)
        x_ts, y_ts = load_data.load_ccat_data(test_path)

        #x_train_rep, y_train_rep, x_test_rep = feature_extractor.feature_extraction(x_train, x_train, x_ts, feature_code)
        #LogReg(x_train_rep, y_train, x_test_rep, y_ts)

        x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        x_train_rep, x_val_rep, x_test_rep = feature_extractor.feature_extraction(x_tr, x_val, x_ts, feature_code)
        FNN(x_train_rep, x_val_rep, x_test_rep, y_tr, y_val, y_ts,
        args.batch_size, args.nb_class, args.nb_epoch, args.lr, args.model_path, args.weight_path, args.hidden_size, args.feature_dim, args.dropout)

    elif data == "judgment":
        file = "/home/yunita/Data/Dataset/judgement/tokenised.txt"
        x, y = load_data.load_judgment(file)
        skf = StratifiedKFold(y, n_folds=10)
        list_acc_logreg = []
        for train_index, test_index in skf:
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            x_train_rep, y_train_rep, x_test_rep, y_test_rep = feature_extractor.feature_extraction(X_train, Y_train, X_test, Y_test, feature_code)
            list_acc_logreg.append(LogReg(x_train_rep, y_train_rep, x_test_rep,  y_test_rep))
        print (np.mean(list_acc_logreg))

    elif data == "imdb":
        file = "/home/yunita/Data/Dataset/imdb62/imdb62post.txt"
        x, y = load_data.load_imdb62(file)
        skf = StratifiedKFold(y, n_folds=10)
        list_acc_logreg = []
        for train_index, test_index in skf:
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            x_train_rep, y_train_rep, x_test_rep, y_test_rep = feature_extractor.feature_extraction(X_train, Y_train, X_test, Y_test, feature_code)
            list_acc_logreg.append(LogReg(x_train_rep, y_train_rep, x_test_rep,  y_test_rep))
        print (np.mean(list_acc_logreg))

if __name__ == '__main__':
    main()