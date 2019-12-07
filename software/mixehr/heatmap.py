import os
import pickle
from sys import platform as sys_pf

import h5py
import numpy as np
import pandas as pd
from plot import plot_topics1

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
if sys_pf == 'linux':
    import matplotlib
import matplotlib.pyplot as plt


def normalize(exp_p):
    n_types, n_words, n_topics = exp_p.shape
    #
    # for w in range(n_words):
    #     for t in range(n_types):
    #         exp_p[t, w, :] /= exp_p[t, w, :].sum()
    #         print(exp_p[t, w, :])
    #
    for k in range(n_topics):
        wt = exp_p[:, :, k]
        norm = np.max(wt.sum(axis=1))
        priority = norm / wt.sum(axis=1)
        for t, priority_t in enumerate(priority):
            exp_p[t, :, k] *= priority_t
    return exp_p


# exp_p_file = '/Users/cuent/Downloads/exp_p_250.pkl'
# p = pickle.load(open(exp_p_file, 'rb'))


def select_unique_top_words(max_words, topics):
    top_words = []
    types = []
    for k in topics:
        word_type = p[:, :, k]
        n_types, n_words = word_type.shape
        # sort arguments (type, word) from matrix
        sort_word_type = np.dstack(np.unravel_index(np.argsort(word_type.ravel())[::-1], (n_types, n_words)))[0]

        selected_words = 0
        for type_id, word_id in sort_word_type:
            if word_id not in top_words:
                top_words.append(word_id)
                types.append(type_id)
                selected_words += 1
            if selected_words == max_words:
                break

    return top_words, types


def select_top_words(max_words, topics):
    top_words = []
    types = []
    for k in topics:
        word_type = p[:, :, k]
        n_types, n_words = word_type.shape
        # sort arguments (type, word) from matrix
        sort_word_type = np.dstack(np.unravel_index(np.argsort(word_type.ravel())[::-1], (n_types, n_words)))[0]

        selected_words = 0
        for type_id, word_id in sort_word_type:
            top_words.append(word_id)
            types.append(type_id)
            selected_words += 1
            if selected_words == max_words:
                break

    return top_words, types


def heat_weights(max_words, topics, unique=True):
    if unique:
        top_words, types = select_unique_top_words(max_words, topics)
    else:
        top_words, types = select_top_words(max_words, topics)
    filtered_words = p[:, top_words, :]
    filtered_words_topics = filtered_words[:, :, topics]

    word_topic_heat = np.zeros((max_words * len(topics), len(topics)))
    for i, t in enumerate(types):
        word_topic_heat[i] = filtered_words_topics[t][i]
    # TODO: get distribution of words over types
    word_type_heat = filtered_words_topics.mean(axis=2).T

    return top_words, types, word_topic_heat, word_type_heat


def parse_type_file(file):
    ids = []
    words = []
    with open(file, 'r') as f:
        lines = f.readlines()
        header = [w.replace('\n', '') for w in lines[0].split(',')]
        for i, l in enumerate(lines[1:]):
            l = l.split(',')
            word = l[-1].replace('\n', '')
            # wordId = int(l[2].replace('\n', ''))
            wordId = i + 1

            ids.append(wordId)
            words.append(word)
    return pd.DataFrame({header[2]: ids, header[-1]: words})


def make_heatmap(max_words, topics, mixehr_dir=None, meta_file=None):
    all_types = list(range(6))
    words, types, heat_words, heat_types = heat_weights(max_words, topics)

    wordids = words
    typeids = types

    if mixehr_dir and meta_file:
        type_vocab, words_vocab = pickle.load(open(meta_file, 'rb'))
        vocab_raw = {
            1: pd.read_csv(os.path.join(mixehr_dir, 'ehrFeatInfo/type1_notes.csv')),
            2: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type2_icd_cm.csv')),
            3: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type3_icd_cpt.csv')),
            4: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type4_lab.csv')),
            5: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type5_presc.csv')),
            6: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type6_drg.csv')),
        }

        all_types_text = {1: 'notes', 2: 'icd-cm', 3: 'icd-cpt', 4: 'lab', 5: "presc", 6: "drg"}

        words_vocab_inv = {}
        type_vocab_inv = {}
        for k, v in words_vocab.items():
            words_vocab_inv[v] = k
        for k, v in type_vocab.items():
            type_vocab_inv[v] = k

        words = [words_vocab_inv[w + 1] for w in words]
        types = [type_vocab_inv[t + 1] for t in types]

        wordids = words
        typeids = types

        all_types = [type_vocab_inv[t + 1] for t in all_types]
        all_types = [all_types_text[t] for t in all_types]

        words_text = {}
        for w, t in zip(words, types):
            all_words_text = vocab_raw[t][['pheId', 'pheName']]
            w_text = all_words_text.loc[all_words_text['pheId'] == w]['pheName'].tolist()
            words_text[w] = w_text[0]
        words = [words_text[w] for w in words]
    print("heat map shape", heat_words.shape)
    # plot_topics(
    #     ["T%d" % t for t in topics],
    #     heat_words,
    #     words,
    #     heat_types,
    #     all_types
    # )

    plot_topics1(
        ["T%d" % t for t in topics],
        heat_words,
        words,
        types,
        all_types
    )

    return wordids, typeids


def export_topics(store_path, K, top_n=5, mixehr_dir=None, meta_file=None):
    all_types = list(range(6))
    topics = list(range(K))
    words, types, heat_words, heat_types = heat_weights(top_n, topics, False)

    # words_ = words
    # types_ = types

    wordids = words
    typeids = types

    if mixehr_dir and meta_file:
        type_vocab, words_vocab = pickle.load(open(meta_file, 'rb'))
        vocab_raw = {
            1: pd.read_csv(os.path.join(mixehr_dir, 'ehrFeatInfo/type1_notes.csv')),
            2: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type2_icd_cm.csv')),
            3: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type3_icd_cpt.csv')),
            4: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type4_lab.csv')),
            5: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type5_presc.csv')),
            6: parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type6_drg.csv')),
        }

        all_types_text = {1: 'notes', 2: 'icd-cm', 3: 'icd-cpt', 4: 'lab', 5: "presc", 6: "drg"}

        words_vocab_inv = {}
        type_vocab_inv = {}
        for k, v in words_vocab.items():
            words_vocab_inv[v] = k
        for k, v in type_vocab.items():
            type_vocab_inv[v] = k

        words = [words_vocab_inv[w + 1] for w in words]
        types = [type_vocab_inv[t + 1] for t in types]

        wordids = words
        typeids = types

        all_types = [type_vocab_inv[t + 1] for t in all_types]
        all_types = [all_types_text[t] for t in all_types]

        words_text = {}
        for w, t in zip(words, types):
            all_words_text = vocab_raw[t][['pheId', 'pheName']]
            w_text = all_words_text.loc[all_words_text['pheId'] == w]['pheName'].tolist()
            words_text[w] = w_text[0]
        words = [words_text[w] for w in words]

    df = pd.DataFrame(columns=["topic", "words", "types"])
    for i, t in enumerate(topics):
        topic_words = []
        topic_types = []
        for w in range(top_n):
            idx = i * top_n + w
            topic_words.append(words[idx])
            topic_types.append(all_types[types[idx] - 1])
            # print(heat_words[idx, t])
            # print(heat_types[idx, types_[idx]])
        df = df.append({
            "topic": t,
            "words": ';'.join(topic_words),
            "types": ';'.join(topic_types)
        }, ignore_index=True)
    df.to_csv(store_path, index=False)


if __name__ == '__main__':
    model_path = "/Users/cuent/Desktop/single/model_mixehr_75_450.hdf5"
    max_words = 5
    storeTo = '/Users/cuent/Desktop/single/'
    show = True
    mixehr_dir = '/Users/cuent/Downloads/processed_new'
    meta_file = '/Users/cuent/Desktop/single/meta.pkl'

    with h5py.File(os.path.join(model_path), 'r') as hf:
        p = hf["exp_p"][...]
        m = hf["m_"][...]

    # get normalized probabilities
    p = normalize(p)
    _, _, K = p.shape
    print("exp p shape", p.shape)

    # get relevant topics
    lbl = np.argsort(m)
    neg_topics = lbl[:max_words]
    pos_topics = list(reversed(lbl[-max_words-2:-2]))

    # topics = pos_topics  # [66, 21, 25, 43, 17, 63]
    print(m)

    for type, topics in zip(["positive", "negative"], [pos_topics, neg_topics]):
        if mixehr_dir and meta_file:
            ws, ts = make_heatmap(max_words, topics, mixehr_dir, meta_file)
        else:
            make_heatmap(max_words, topics)
        # plt.title("Most predictive topics" if type == "positive" else "Least predictive topics")
        if storeTo:
            plt.savefig(os.path.join(storeTo, "%s_k%d.jpg" % (type, K)), transparent=True, optimize=True)
        if show:
            plt.show()
    if mixehr_dir and meta_file and storeTo:
        export_topics(os.path.join(storeTo, "top_topic_word_type_K75.csv"), K=K, top_n=max_words, mixehr_dir=mixehr_dir,
                      meta_file=meta_file)
