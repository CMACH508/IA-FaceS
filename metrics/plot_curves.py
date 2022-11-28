import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.ticker import (AutoMinorLocator)

color = ["darkorange", 'darkorange', 'lime', 'lime']


def read_data(file_name):
    all_data = []
    with open(file_name, 'r') as f:
        datas = f.readlines()
    for data in datas:
        all_data.append(float(data.strip()))
    return all_data


dicts = {
    "bushy": {
        "y-range": [-0.025, 0.01],
        "x-range": [0, 1]
    },
    "beard": {
        "y-range": [-0.1, 0.03],
        "x-range": [0, 1]
    }

}

dicts2 = {
    "bushy": {
        "y-range": [0, 0.1],
        "x-range": [0, 1]
    },
    "beard": {
        "y-range": [0, 0.1],
        "x-range": [0, 1]
    }

}


def plot_attr_accuracy(metric, data_dir, attr, colors, name, marker):
    """
    Figure 3(c)
    :param data_dir:
    :param data_set:
    :param metric:
    :return:
    """
    data_dir = os.path.join(data_dir, attr)
    if attr == 'beard':
        acc = (1000 - np.array(read_data(data_dir + '/acc.txt'))) / 1000
    else:
        acc = np.array(read_data(data_dir + '/acc.txt')) / 1000

    d_score = np.array(read_data(data_dir + '/d_score_drop.txt'))

    mse = np.array(read_data(data_dir + '/mse_score.txt'))
    x = acc

    if metric == 'mse':
        y = mse
    else:
        y = d_score

    plot(x, y, color=colors[0], linewidth=2.5, linestyle="--", marker=marker, markersize=7,
         label=name)
    plt.fill_between(x, 0, y, facecolor=colors[1], alpha=0.3)
    return


def plot_curves(attr, metric):
    data_dir1 = 'data/inter-curve-score/our_data'
    data_dir2 = 'data/inter-curve-score/stylegan2_data'

    plot_attr_accuracy(metric, data_dir1, attr, color[0:2], 'IA-FaceS (Ours)', marker='o')
    plot_attr_accuracy(metric, data_dir2, attr, color[2:4], 'SyleGAN2', marker='s')

    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18

    }
    if metric == 'mse':
        x_range = dicts2[attr]['x-range']
        y_range = dicts2[attr]['y-range']
    else:
        x_range = dicts[attr]['x-range']
        y_range = dicts[attr]['y-range']

    plt.xlim(x_range[0], x_range[1], font1)
    plt.ylim(y_range[0], y_range[1], font1)

    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='both', alpha=0.5, linestyle='--')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('axes', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel('Manipulation Accuracy', font1)
    if metric == 'mse':
        plt.ylabel('Change of Irrelevant Regions', font1)
    else:
        plt.ylabel('Image Fidelity Gap', font1)

    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15

    }
    ax.legend(loc='upper left', prop=font2)

    if attr == 'bushy':
        legend2 = 'Bushy eyebrows'
    else:
        legend2 = 'Beard'

    if metric == 'mse':
        plt.text(0.1, 0.04, legend2, family='Times New Roman', fontsize=16, weight='heavy', style='normal',
                 bbox=dict(boxstyle='square,pad=0.5', fc='none', ec='black', lw=1, alpha=0.7))
    else:
        plt.text(0.1, -0.08, legend2, family='Times New Roman', fontsize=16, weight='heavy', style='normal',
                 bbox=dict(boxstyle='square,pad=0.5', fc='none', ec='black', lw=1, alpha=0.7))

    plt.show()

    return


if __name__ == "__main__":
    plot_curves('bushy', 'mse')
