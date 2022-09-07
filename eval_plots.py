#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : eval_plots.py
#   Author      : ronen halevy 
#   Created date:  9/7/22
#   Description :
#
# ================================================================


import matplotlib.pyplot as plt
import numpy as np

# Set the figure size
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True


def barh_multiple_plots(ypos, ylables, vals_lst,  width, shift_bar, xlim, colors, opacity, title, xlabel, legend):
    fig, ax = plt.subplots()
    barhs = [ax.barh(y_pos+shift, vals, width,  color=color, alpha=opacity) for idx, (vals, shift, color) in enumerate(zip(vals_lst, shift_bar, colors))]
    ax.set_yticks(y_pos, labels=ylables)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    [ax.bar_label(barhs[idx], fmt='%.2f') for idx in range(len(barhs))]
    ax.set_xlim(right=xlim)
    ax.legend(labels=legend)
    plt.show()


if __name__ == '__main__':
    tp = [5, 6, 7, 8, 9, 10, 11, 62]
    fp = [3, 4, 5, 6, 7, 8, 9, 33]
    fn = [2, 14, 15, 16, 17, 18, 1, 3]
    gt = [3, 4, 5, 6, 7, 8, 9, 35]
    pred = [13, 14, 2, 16, 4, 8, 9, 35]

    title = 'title'
    xlabel = 'Performance'
    ylables = ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8')
    legend = ylables
    vals_lst = [tp, fp,fn,gt,pred]
    shift_bar = [0,0,0,1,2]
    xlim = np.amax(vals_lst) * 1.1
    colors = ['red', 'blue', 'green', 'orange', 'black']
    opacity = 0.5
    gap = 0.1
    width = (1. - gap)
    y_pos = np.arange(len(ylables))*(np.count_nonzero(shift_bar)+1)

    barh_multiple_plots(y_pos, ylables, vals_lst, width, shift_bar, xlim, colors, opacity, title, xlabel, legend)

