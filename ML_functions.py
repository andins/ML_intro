#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:58:34 2018

@author: andrea insabato a.insabato@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as spl
from sklearn.model_selection import KFold
import seaborn as sns


def plot_error_surface(x, t):
    w1 = np.linspace(-100, 100, 100)
    w2 = np.linspace(-100, 100, 120)
    Esurf = np.zeros([len(w1), len(w2)])
    for i in range(len(w1)):
        for j in range(len(w2)):
            wfig = np.reshape(np.array([w1[i], w2[j]]), [2, 1])
            Esurf[i, j] = np.log(error_fnc(wfig, x, t))

    Emin = np.min(Esurf)
    xmin, ymin = np.where(Esurf == Emin)
    plt.figure()
    seamap = mpl.colors.ListedColormap(sns.color_palette("Blues", 256))
    plt.contourf(w2, w1, Esurf, 20, cmap=seamap)
    plt.colorbar(label='log Error')
    plt.plot(w2[ymin], w1[xmin], '+r')
    plt.xlabel(r'$w_1$')
    plt.ylabel(r'$w_0$')
    return np.array([xmin, ymin])


def error_fnc(w, x, t):
    return np.mean((np.dot(w.T, x) - t)**2)/2


def split_train_test(x, t, k=3):
    kf = KFold(n_splits=k)
    x_train = dict()
    x_test = dict()
    t_train = dict()
    t_test = dict()
    i = 0
    for train, test in kf.split(x):
        x_train[i] = x[train]
        x_test[i] = x[test]
        t_train[i] = t[train]
        t_test[i] = t[test]
        i += 1
    return x_train, x_test, t_train, t_test


def gradient_descent(x, t):
    plot_error_surface(x, t)
    w = np.zeros([2, 10])
    init = np.random.randn(2)*100
    # draw initial conditions inside the figure but far from minimum
    while (np.any(np.abs(init) > 100) or spl.norm(init) < 80):
        init = np.random.randn(2)*100
    w[:, 0] = init

    for s in range(9):
        w_now = w[:, s]
        grad = np.mean(np.repeat(np.reshape((np.dot(w_now.T, x) - t), [1, x.shape[1]]), 1, axis=0) * x, axis=1)
        w[:, s+1] = w_now - grad
        plt.arrow(w_now[0], w_now[1], -grad[0], -grad[1],
                  head_width=2, head_length=3, fc='r', ec='r',
                  length_includes_head=True,
                  shape='full')
    plt.scatter(w[0, :], w[1, :], color='yellow')


class supermodel:
    def __init__(self, deg=10):
        self.deg = deg
        self.w = np.random.randn(self.deg)

    def fit(self, X, t):
        self.w = np.polyfit(X, t, self.deg)

    def predict(self, X):
        y = np.polyval(self.w, X)
        return y

    def error(self, X, t):
        error = np.mean((self.predict(X) - t)**2)
        return error

x = np.random.rand(1000)
x1 = np.stack([np.ones([1000]), x])
t = 1 + x*2 + np.random.randn(1000) * .5
wmin = plot_error_surface(x1, t)