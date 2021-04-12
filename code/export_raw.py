from code import Corpus
import pandas as pd
from tqdm import tqdm
import pickle
import os


def extract_docs(corpus_dir, vocab, out_dir):
    corpus, meta = Corpus.read_corpus_from_directory(corpus_dir, True)

    type_ids, vocab_ids = meta
    type_ids_rev = {}
    vocab_ids_rev = {}

    for k in type_ids:
        type_ids_rev[type_ids[k]] = k

    for k in vocab_ids:
        vocab_ids_rev[vocab_ids[k]] = k

    ignored_words = 0
    docs_only = []
    docs_freq = []
    patients = []
    ids = []
    responses = []

    pbar = tqdm(corpus)
    for c, _ in pbar:
        patient_id = c.index
        i = c.patient_id
        label = c.y
        words = c.words_dict
        flat_words = []
        flat_words_freq = []
        for (type_id, word_id), freq in words.items():
            type_id = type_ids_rev[type_id]
            word_id = vocab_ids_rev[word_id]

            flat_words_freq.append("%d:%d" % (word_id, freq))

            vocab_type = vocab[type_id][['pheId', 'pheName']]
            w = vocab_type.loc[vocab_type['pheId'] == word_id]['pheName'].tolist()
            if len(w) > 1:
                print(w)
            w = freq * w
            if len(w) > 0:
                flat_words.extend(w)
            else:
                print(type_id, word_id)
                ignored_words += 1
        docs_only.append(' '.join(flat_words))
        docs_freq.append("%d %s" % (len(flat_words_freq), ' '.join(flat_words_freq)))
        ids.append(i)
        patients.append(patient_id)
        responses.append(label)
    if ignored_words > 0:
        print("Couldn't find %d words." % ignored_words)

    data = {'mixehr_id': ids, 'patient_id': patients, 'label': responses, 'text': docs_only}

    # save data
    mixehr_data = pd.DataFrame(data)
    mixehr_data.to_csv(os.path.join(out_dir, 'mix_raw.csv'), index=False)
    # save labels only
    mixehr_data[['label']].to_csv(os.path.join(out_dir, 'mix_label.csv'), index=False, header=False)
    # save slda format
    pd.DataFrame({'text': docs_freq}).to_csv(os.path.join(out_dir, 'mix_word_freq.csv'), index=False, header=False)
    # save vocabulary
    pickle.dump(vocab, open(os.path.join(out_dir, 'vocab.pkl'), 'wb'))
    # save id lookup
    pickle.dump((type_ids, vocab_ids), open(os.path.join(out_dir, 'id_mixehr_seq.pkl'), 'wb'))
    pickle.dump((type_ids_rev, vocab_ids_rev), open(os.path.join(out_dir, 'id_seq_mixehr.pkl'), 'wb'))


def __parse_type_file(file):
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


def __parse_exchange_cols(file):
    df = pd.read_csv(file, header=None)
    words = df[0].values
    ids = df[1].values
    return pd.DataFrame({"pheId": ids, "pheName": words})


if __name__ == '__main__':
    out_dir = '/Users/cuent/Downloads/processed_new/mv/out/test'
    corpus_dir = '/Users/cuent/Downloads/processed_new/mv/out/test'
    mixehr_dir = '/Users/cuent/Downloads/processed_new/mv'

    # vocab = {
    #     1: pd.read_csv(os.path.join(mixehr_dir, 'ehrFeatInfo/type1_notes.csv')),
    #     2: __parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type2_icd_cm.csv')),
    #     3: __parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type3_icd_cpt.csv')),
    #     4: __parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type4_lab.csv')),
    #     5: __parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type5_presc.csv')),
    #     6: __parse_type_file(os.path.join(mixehr_dir, 'ehrFeatInfo/type6_drg.csv')),
    # }
    vocab = {
        1: __parse_exchange_cols(os.path.join(mixehr_dir, '1_vocab.txt')),
        2: __parse_exchange_cols(os.path.join(mixehr_dir, '2_vocab.txt')),
    }

    extract_docs(corpus_dir, vocab, out_dir)
