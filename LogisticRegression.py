#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from scipy.optimize import check_grad

# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc = 2):
    if data_name == 'synthetic-easy':
        print 'Creating easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print 'Creating medium synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print 'Creating hard easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print 'Creating two moons dataset'
        X, y = datasets.make_moons(noise=0.2, random_state=0)
    elif data_name == 'circles':
        print 'Creating two circles dataset'
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print 'Loading iris dataset'
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print 'Loading digits dataset'
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print 'Loading breast cancer dataset'
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print 'Cannot find the requested data_name'
        assert False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# compare the prediction with grount-truth, evaluate the score
def myscore(y, y_gt):
    assert len(y) ==  len(y_gt)
    return np.sum(y == y_gt)/float(len(y))

# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

####################################################
# binary label classification

def sig(z):
    return 1.0/(1.0 + np.exp(-z))

# x is 1*k, w is k * 1
def hypo(x, w):
    return sig(np.dot(x, w))

def one_hot(Y, k):
    n = Y.shape[0]
    ret = np.zeros((n, k))
    ret[np.arange(n), Y] = 1
    return ret

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def pred(x, w):
    return softmax(np.dot(x, w))

# X is N*(k+1), Y is N*1, w is (k+1) * 1
def cost_function(X, Y, w):
    y_hat = hypo(X, w)
    n = X.shape[0]
    return -np.mean((1-Y) * np.log(1-y_hat) + Y * np.log(y_hat))

# X is N*(k+1), Y is N*1, w is (k+1) * 1
def cost_function_grad(X, Y, w):
    y_hat = hypo(X, w)
    n = X.shape[0]
    return 1.0/n * np.dot(X.T, (y_hat - Y))

# X is N*(k+1), Y is N*c, w is (k+1) * c
def cost_function_multi(X, Y, w):
    y_hat = pred(X, w) # N * c
    n = X.shape[0]
    return -1.0 / n * np.sum(Y * np.log(y_hat))

# X is N*(k+1), Y is N*c, w is (k+1) * c
def cost_function_multi_grad(X, Y, w):
    y_hat = pred(X, w)
    n = X.shape[0]
    return 1.0/n * np.dot(X.T, (y_hat - Y))

def gradient_descent(X, Y, w, a):
    return w - a * cost_function_grad(X, Y, w)

def gradient_descent_multi(X, Y, w, a):
    return w - a * cost_function_multi_grad(X, Y, w)

# train the weight vector w
def mytrain_binary(X_train, y_train):
    print 'Start training ...'
    a = 0.05
    X_extended = np.hstack([X_train, np.ones([X_train.shape[0],1])])
    nfeatures = X_extended.shape[1]
    w = np.zeros(nfeatures)
    prev_cost = 1
    cost_list = []
    for x in xrange(5000):
        new_w = gradient_descent(X_extended, y_train, w, a)
        cost = cost_function(X_extended, y_train, new_w)
        if cost < prev_cost:
            cost_list.append(cost)
            w = new_w
            a = 1.01 * a
            if prev_cost - cost < 1e-9:
                break
            prev_cost = cost
        else:
            a = 0.5 * a
        
    print 'Finished training.'
    #plt.plot(cost_list)
    #plt.show()
    return w

# compute y on any input set X, and current weights w
# returns a list of length N, each entry is between 0 and 1
def mypredict_binary(X, w):
    X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    return hypo(X_extended, w)

# predict labels using the logistic regression model on any input set X, using current w
# returns a list of length N, each entry is either 0 or 1
def mytest_binary(X, w):

    # here is a fake implementation, you should replace it
    assert len(w) == X.shape[1] + 1
    w_vec = np.reshape(w,(-1,1))
    X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    y_pred = np.ravel(np.sign(np.dot(X_extended,w_vec)))
    y_pred_final = np.maximum(np.zeros(y_pred.shape), y_pred)
    return y_pred_final

# draw results on 2D plan for binary classification
# use it for debugging
def draw_result_binary(X_train, X_test, y_train, y_test, w):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.figure(1)
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    # Put the result into a color plot
    tmpX = np.c_[xx.ravel(), yy.ravel()]
    Z = mypredict_binary(tmpX, w)
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha = .4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend()

    y_predict = mypredict_binary(X_test,w)
    score = myscore(y_predict, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    plt.show()

####################
# multi-label classification

# train the weight vector w
def mytrain_multi(X_train, y_train):
    print 'Start training ...'

    a = 0.05
    X_extended = np.hstack([X_train, np.ones([X_train.shape[0],1])])
    nfeatures = X_extended.shape[1]
    nclasses = len(np.unique(y_train))
    w = np.zeros([nfeatures, nclasses])
    y_train = one_hot(y_train, nclasses)
    prev_cost = nclasses
    cost_list = []
    for x in xrange(5000):
        new_w = gradient_descent_multi(X_extended, y_train, w, a)
        cost = cost_function_multi(X_extended, y_train, new_w)
        if cost < prev_cost:
            cost_list.append(cost)
            w = new_w
            a = 1.01 * a
            if prev_cost - cost < 1e-9:
                break
            prev_cost = cost
        else:
            a = 0.5 * a

    print 'Finished training.'
    #plt.plot(cost_list)
    #plt.show()
    return w

# compute y, NxK matrix, values between 0 and 1
def mypredict_multi(X, w):
    X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    return pred(X_extended, w)

# predict labels using the logistic regression model on any input set X
# return length N list, value between 0 and K-1
def mytest_multi(X, w):
    y_pred = mypredict_multi(X, w)
    y_pred_final = np.argmax(y_pred, axis=1)
    return y_pred_final

################

def main():

    #######################
    # get data
    # binary labeled

    #X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    #X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    #X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    #X_train, X_test, y_train, y_test = acquire_data('moons')
    #X_train, X_test, y_train, y_test = acquire_data('circles')
    #X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    # multi-labeled
    #X_train, X_test, y_train, y_test = acquire_data('synthetic-easy', nc = 3)
    #X_train, X_test, y_train, y_test = acquire_data('synthetic-medium', nc = 3)
    X_train, X_test, y_train, y_test = acquire_data('synthetic-hard', nc = 3)
    #X_train, X_test, y_train, y_test = acquire_data('iris')
    #X_train, X_test, y_train, y_test = acquire_data('digits')


    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    #draw_data(X_train, X_test, y_train, y_test, nclasses)
    if nclasses == 2:
        w_opt = mytrain_binary(X_train, y_train)
        # debugging example
        #draw_result_binary(X_train, X_test, y_train, y_test, w_opt)
    else:
        w_opt = mytrain_multi(X_train, y_train)

    if nclasses == 2:
        y_train_pred = mytest_binary(X_train, w_opt)
        y_test_pred = mytest_binary(X_test, w_opt)
    else:
        y_train_pred = mytest_multi(X_train, w_opt)
        y_test_pred = mytest_multi(X_test, w_opt)

    train_score = myscore(y_train_pred, y_train)
    test_score = myscore(y_test_pred, y_test)

    print 'Training Score:', train_score
    print 'Test Score:', test_score

if __name__ == "__main__": main()