from sys import platform as sys_pf

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
if sys_pf == 'linux':
    import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np


def plot_topics(topic_list, m, word_list, y=None, type_list=None):
    '''
      m: word-topic probabilities (W,K)
      y: types (W,T)
      word_list: list of word labels (W)
      topic_list: list of topic labels (K)
      type_list: list of type labels (W)
    '''
    nwords, ntopics = m.shape

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    grid = dict(height_ratios=[0.5, m.shape[0]], width_ratios=[m.shape[1], y.shape[1] * 0.08])
    fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw=grid, figsize=(15, 20))

    # images
    axes[1, 0].imshow(m, aspect="auto", cmap="Reds")  # , norm=norm)
    if not isinstance(y, np.ndarray):  # y == None or type_list == None:
        axes[1, 1].axis("off")
    else:
        axes[1, 1].imshow(y, aspect="auto", cmap="Blues")  # , norm=norm)
    axes[0, 1].axis("off")

    # labels y-axis - words
    axes[1, 0].set_yticks(np.arange(nwords), minor=False)
    axes[1, 0].set_yticklabels(word_list, fontdict=None, minor=False)

    # labels x-axis - topics
    axes[1, 0].set_xticks(np.arange(ntopics), minor=False)
    axes[1, 0].set_xticklabels(topic_list, fontdict=None, minor=False, rotation='90')

    # labels y-axis - types
    if type_list != None:
        axes[1, 1].set_xticks(np.arange(len(type_list)), minor=False)
        axes[1, 1].set_xticklabels(type_list, fontdict=None, minor=False, rotation='90')
        # axes[1, 1].yaxis.tick_right()
        axes[1, 1].set_xlabel('Types')

    # titles
    axes[1, 0].set_xlabel('Topics')
    axes[1, 0].set_ylabel('Words')

    for ax in [axes[1, 1]]:
        ax.set_yticks([])  # ; ax.set_yticks([])

    sm = matplotlib.cm.ScalarMappable(cmap="Reds")  # , norm=norm)
    sm.set_array([])

    fig.colorbar(sm, cax=axes[0, 0], orientation='horizontal')
    plt.show()


def plot_topics1(topic_list, m, word_list, y=None, type_list=None):
    '''
      m: word-topic probabilities (W,K)
      y: types (W)
      word_list: list of word labels (W)
      topic_list: list of topic labels (K)
      type_list: list of type labels (W)
    '''
    # color types
    color_types = ['g', 'b', 'r', 'c', 'm', 'y']
    cmap = colors.ListedColormap(color_types)
    norm = colors.BoundaryNorm([1, 2, 3, 4, 5, 6, 7], cmap.N)

    nwords, ntopics = m.shape

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    grid = dict(height_ratios=[0.5, m.shape[0]], width_ratios=[.05 * m.shape[1], 1])
    fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw=grid, figsize=(15, 20))

    # images
    axes[1, 0].imshow(m, aspect="auto", cmap="Reds")  # , norm=norm)
    if not y:  # y == None or type_list == None:
        axes[1, 1].axis("off")
    else:
        y = np.array(y)[:, np.newaxis]
        axes[1, 1].imshow(y, cmap=cmap, norm=norm)
    axes[0, 1].axis("off")

    # limit length of words
    word_list = [str(w) for w in word_list]
    word_list = [w[:60] + str("..." if len(w) > 60 else "") for w in word_list]
    # labels y-axis - words
    axes[1, 0].set_yticks(np.arange(nwords), minor=False)
    axes[1, 0].set_yticklabels(word_list, fontsize=8, fontdict=None, minor=False)

    # labels x-axis - topics
    axes[1, 0].set_xticks(np.arange(ntopics), minor=False)
    axes[1, 0].set_xticklabels(topic_list, fontdict=None, minor=False, rotation='0')

    # labels y-axis - types
    if type_list != None:
        axes[1, 1].set_xticks([])
        patches = []
        for col, lbl in zip(color_types, type_list):
            patches.append(mpatches.Patch(color=col, label=lbl))
        axes[1, 1].legend(handles=patches, loc='center left', bbox_to_anchor=(1., 0.5), title="Types")
        # dot useful to position legend
        # axes[1, 1].scatter((1.), (0.5), s=81, c="limegreen", transform=axes[1, 1].transAxes)

    # titles
    axes[1, 0].set_xlabel('Topics')
    axes[1, 0].set_ylabel('Words')

    for ax in [axes[1, 1]]:
        ax.set_yticks([])  # ; ax.set_yticks([])

    sm = matplotlib.cm.ScalarMappable(cmap="Reds")  # , norm=norm)
    sm.set_array([])
    plt.subplots_adjust(wspace=-.76, left=0.3, right=1.9)
    cb = fig.colorbar(sm, cax=axes[0, 0], orientation='horizontal')
    # cb.yaxis.set_label_position('top')
    # cb.yaxis.set_ticks_position('top')
    # plt.tight_layout()
    # plt.show()


def select_words(n, num_wordspertopic):
    # n: word-topic (K,W)
    # num_wordspertopic: max words per topic to extract

    # select top words per topic
    top_word_idx = n.argsort(axis=1)[:, -num_wordspertopic:]
    low_word_idx = n.argsort(axis=1)[:, :num_wordspertopic]

    # flatten indexes and remove repeated words
    top_word_idx = top_word_idx.flatten()
    low_word_idx = low_word_idx.flatten()

    top_prob = n[:, top_word_idx]
    low_prob = n[:, low_word_idx]

    return top_prob, low_prob, top_word_idx, low_word_idx


def plot_coefficients(coefficients):
    mortality_linear_coefficients = coefficients
    mortality_topic_coeffcients = coefficients.argsort()

    y = mortality_linear_coefficients[mortality_topic_coeffcients]
    x = list(range(y.shape[0]))

    # plot coefficients
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(x, y)
    ax.plot(y)

    # labels
    ax.set_ylabel('Mortality topic coefficients')
    ax.set_xlabel('Linear coefficients for mortality')

    # add topic labels
    for i, topic in enumerate(mortality_topic_coeffcients):
        ax.annotate(topic, (x[i], y[i]))

    plt.show()
