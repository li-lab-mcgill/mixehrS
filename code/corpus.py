from typing import Mapping, List, NoReturn, Set, TypeVar
import numpy as np
import pandas as pd
import pickle
import os
import logging
import sys
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from utils import generate_data


P = TypeVar('P')

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Select one command', dest='cmd')

# parser process
parser_process = subparsers.add_parser('process', help="Transform MixEHR raw data")
parser_process.add_argument("-im", "--ignore_missing", help="Ignores observations with missing values",
                            action='store_true', default=False)
parser_process.add_argument("-n", "--max", help="Maximum number of observations to select", type=int, default=None)

# parser process
parser_split = subparsers.add_parser('split', help="Split data into train/test")
parser_split.add_argument("-tr", "--testing_rate", help="Testing rate. Default: 0.2", type=float, default=0.2)

# parser filter
parser_filter = subparsers.add_parser('filter', help="Select only observations with a target")

# parser process
parser_skf = subparsers.add_parser('stratifiedcv', help="Stratified K-Folds cross-validator")
parser_skf.add_argument("-cv", "--n_splits", help="Number of folds", type=int, default=2)

# default arguments
parser.add_argument('input', help='Directory containing input data')
parser.add_argument('output', help='Directory where processed data will be stored')


class Corpus(Dataset):
    def __init__(self, patients: List[P], T: int, W: int, C: int) -> NoReturn:
        logger.info("Creating corpus...")
        self.dataset = patients
        # self.M = M
        self.D = len(patients)

        self.T = T
        self.W = W
        self.C = C

    def __len__(self):
        return self.D

    def __getitem__(self, index):
        '''
        Generate one sample
        :param index:
        :return:
        '''
        patient_sample = self.dataset[index]

        word_type_freq = patient_sample.words_dict
        type_freq = {}
        word_freq = {}

        for pair in word_type_freq:
            type, word = pair
            freq = word_type_freq[pair]
            if type not in type_freq:
                type_freq[type] = 0

            if word not in word_freq:
                word_freq[word] = 0

            type_freq[type] += freq
            word_freq[word] += freq

        patient_sample.word_freq = word_freq
        patient_sample.type_freq = type_freq

        return patient_sample, index

    @staticmethod
    def build_corpus_sample(K, T, W, D, M, a=1, b=1, tau=4):
        '''
        Builds a toy Corpus dataset.
        Generates toy data using the generative model of MixEHR.
        :param K: number of topics
        :param T: number of types
        :param W: number of words in the vocabulary
        :param D: number of documents
        :param M: number of words. We assume that all documents have same length.
        :param a: shape parameter of gamma distribution; used to sample the hyper-parameters
        :param b: scale parameter of gamma distribution; used to sample the hyper-parameters
        :param tau: mean used to sample w
        :return: y: response b: types x: words z: topics-assignment g: response
        '''
        y, b, x, z, g, theta = generate_data(K, T, W, D, M, a, b, tau)
        dataset = []
        C = 0
        for i in range(D):
            cnt = Counter()
            len(b[:, i])
            patient = Corpus.Patient(i, i, y[i])
            for batch in zip(b[:, i], x[:, i]):
                cnt[batch] += 17
            for type, word in cnt:
                freq = cnt[(type, word)]
                patient.append_record(type, word, freq)
                C += freq
            dataset.append(patient)
        corpus = Corpus(dataset, T, W, C)
        corpus.z = z
        corpus.g = g

        return corpus, theta

    @staticmethod
    def __collate_mixehr__(batch):
        '''
        Returns a batch for each iteration of the DataLoader
        :param batch:
        :return:
        '''
        patients, indixes = zip(*batch)
        return list(patients), np.array(indixes), np.sum([p[0].Cj for p in batch])

    @staticmethod
    def build_from_mixehr_fileformat(data_path: str, label_path: str, persist_path: str = None,
                                     ignore_missing_labels: bool = False, top_n: int = None):
    # def build_from_mixehr_fileformat(data_path: str, meta_path: str, label_path: str, persist_path: str = None,
    #                                  ignore_missing_labels: bool = False, top_n: int = None):
        '''
        Reads a datafiles in the mixehr format and returns a Corpus object.
        :param data_path: data records, no header, columns are separated with spaces.
                        It contains:
                            SUBJECT_ID (i.e., patient ID), data type ID, variable ID under the data type, frequency.
        :param meta_path: summary of the total number of variable under each of the data types. Separation: space, not header.
                            SUBJECT_ID, total
        :param label_path: For each SUBJECT_ID, whether the patient died at the last admission
                            0 indicates patients are still alive in the last admission
                            1 indicates patients are dead in the last admission but obviously alive at early admission

                            Headers:
                            SUBJECT_ID,HOSPITAL_EXPIRE_FLAG

                            Separation: space

                            Note: ‘NA’ indicates the patient only has one admission (non-applicable).
                            The supervision module will need ignore these patients (i.e., not taking into account the
                            corresponding predictive likelihood). It treats as missing data; missing
                            data is represented with -1.
        :param persist_path: if specified, dumps the read data in disk. It stores two files:
                            corpus.pkl: Corpus object used to train MIXEHR model
                            meta.pkl: it's a set with all the ids for patient_ids, type_ids, vocab_ids
        :return: Corpus object.
        '''

        # Read response
        def __read_response__(patient_ids, labels):
            num_patients = len(patient_ids.keys())
            y = {}
            pbar = tqdm(patient_ids.keys())
            for i, patient in enumerate(pbar):
                response = labels[labels['nam'] == patient]['label'].item()
                record = patient_ids[patient]
                y[patient] = response
                pbar.set_description("%.4f  - patient(%s)" % (100 * (i + 1) / num_patients, record))
            return y

        # Read patients
        def __read_patients__(data, y, patient_ids, type_ids, vocab_ids):
            training = {}

            number_records = data.shape[0]
            with tqdm(total=number_records) as pbar:
                C = 0
                for i, row in enumerate(data.iterrows()):
                    row = row[1]
                    patient_id = row['nam']
                    if patient_id not in patient_ids:
                        continue
                    index = patient_ids[patient_id]
                    type_id = type_ids[row['specpro']]
                    word_id = vocab_ids[row['dx']]
                    freq = row['freq']
                    if index not in training:
                        training[index] = Corpus.Patient(index, patient_id, y[patient_id],
                                                         isMissingLabel=y[patient_id] == -1)
                    patient = training[index]
                    C += freq
                    patient.append_record(type_id, word_id, freq)

                    pbar.set_description("%.4f  - patient(%s), type(%s), word(%s)"
                                         % (100 * (i + 1) / number_records, index, type_id, word_id))
                    pbar.update(1)
            return training, C

        def __store_data__(toStore, corpus, metadata):
            print('store data....')
            if not os.path.exists(toStore):
                os.makedirs(toStore)

            corpus_file = os.path.join(toStore, "corpus.pkl")
            # patients_file = os.path.join(toStore, "patients.pkl")
            metadata_file = os.path.join(toStore, "meta.pkl")

            logger.info("Saving: \n\t%s\n\t%s" % (corpus_file, metadata_file))

            pickle.dump(corpus, open(corpus_file, "wb"))
            # pickle.dump(patient_ids, open(patients_file, "wb"))
            pickle.dump(metadata, open(metadata_file, "wb"))

            logger.info("Data stored in %s" % toStore)
            print('finish')

        # read files
        data = pd.read_csv(data_path, sep='\t') # data_RAMQ.txt first 10000 rows
        # meta = pd.read_csv(meta_path, header=None, sep=' ')
        labels = pd.read_csv(label_path) # if read sample files, the path is data_label_sample.csv
        print(data)
        print(labels)

        # M = data.groupby(0).sum()[3].values  # number of words in for record j
        # D = data[0].unique().shape[0]  # number of patients

        # train_size = int(split_rate * D)
        # test_size = D - train_size
        # logger.info("\ntrain size(%s)\ntest_size(%s)" % (train_size, test_size))

        # map data ids
        patient_ids = {}
        # patient_ids_test = {}
        type_ids = {}
        vocab_ids = {}

        # Add sequence to each of the ids
        ids = labels['nam'].unique()
        for i, patient_id in enumerate(ids):
            patient_ids[patient_id] = i
        data = data[data['nam'].isin(patient_ids.keys())]
        for i, type_id in enumerate(data['specpro'].unique()):
            type_ids[type_id] = i # the type is from 0 to start # + 1
        for i, word_type in enumerate(data['dx'].unique()):
            vocab_ids[word_type] = i # the word is from 0 to start # + 1
        with open('mapping/patient_ids.pkl', 'wb') as handle:
            pickle.dump(patient_ids, handle)
        with open('mapping/type_ids.pkl', 'wb') as handle:
            pickle.dump(type_ids, handle)
        with open('mapping/vocab_ids.pkl', 'wb') as handle:
            pickle.dump(vocab_ids, handle)
        print("finish exporting")

        # vocabulary_type_size = meta[1].values
        # W = np.max(vocabulary_type_size)

        # Process and read types, words, and response
        y = __read_response__(patient_ids, labels)
        dataset, C = __read_patients__(data, y, patient_ids, type_ids, vocab_ids)

        # Set data to Corpus object
        T = len(type_ids)
        W = len(vocab_ids)
        corpus = Corpus([*dataset.values()], T, W, C)

        logger.info(f'''
        ========= DataSet Information =========
        Patients: {len(corpus.dataset)}
        Types: {corpus.T}
        Word Tokes: {corpus.W}
        Number words in Corpus: {corpus.C}
        ======================================= 
        ''')
        if persist_path:
            __store_data__(persist_path, corpus, (type_ids, vocab_ids))

        return corpus, type_ids, vocab_ids

    @staticmethod
    def split_train_test(corpus, split_rate: float, toStore: str):
        assert split_rate >= .0 and split_rate <= 1., "specify the rate for splitting training and test. e.g 0.8 = 80% for testing"

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)

            corpus_file = os.path.join(toStore, "corpus.pkl")

            logger.info("Saving: \n\t%s" % (corpus_file))

            pickle.dump(corpus, open(corpus_file, "wb"))

            logger.info("Data stored in %s" % toStore)

        def __split__(size, corpus):
            train_patients = []
            test_patients = []
            corpus_list = [None, None]
            splitted = False

            C = 0
            index = -1

            pbar = tqdm(corpus)
            train_positive_limit = 0.5 * size
            train_negative_limit = size - train_positive_limit
            train_pat_index = []
            train_labels = []
            for p, _, in pbar:
                index += 1
                p.patient_id = index
                pat_label = p.y
                if pat_label == 1 and train_positive_limit >= 0:
                    train_pat_index.append(index)
                    train_positive_limit -= 1
                    train_labels.append(pat_label)
                elif pat_label == 0 and train_negative_limit >= 0:
                    train_pat_index.append(index)
                    train_negative_limit -= 1
                    train_labels.append(pat_label)
                else:
                    continue
            index = -1
            train_index = -1
            test_index = -1
            for p, _, in pbar:
                pbar.set_description("Processing patient %s (index: %s)" % (p.index, p.patient_id))
                index += 1
                if p.patient_id in train_pat_index:
                    train_index += 1
                    p.patient_id = train_index
                    C += p.Cj
                    train_patients.append(p)
                else:
                    test_index += 1
                    p.patient_id = test_index
                    C += p.Cj
                    test_patients.append(p)

                # if index + 1 == size and not splitted:
                #     corpus_list[0] = Corpus(patients, corpus.T, corpus.W, C)
                #     index = -1
                #     C = 0
                #     patients = []
                #     splitted = True
            # print(train_patients)
            # import time
            # time.sleep(100)
            corpus_list[0] = Corpus(train_patients, corpus.T, corpus.W, C)
            corpus_list[1] = Corpus(test_patients, corpus.T, corpus.W, C)

            return tuple(corpus_list)

        train_size = corpus.D - int(split_rate * corpus.D)
        train, test = __split__(train_size, corpus)

        # store data
        __store_data__(os.path.join(toStore, 'test'), test)
        __store_data__(os.path.join(toStore, 'train'), train)

        logger.info("Training size: %s\nTesting size: %s\n" % (train_size, corpus.D - train_size))

    @staticmethod
    def split_stratified_cv(corpus, cv: int, toStore: str):
        assert cv > 1, "Number of folds should be greater than 1"

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)

            corpus_file = os.path.join(toStore, "corpus.pkl")

            logger.info("Saving: \n\t%s" % (corpus_file))

            pickle.dump(corpus, open(corpus_file, "wb"))

            logger.info("Data stored in %s" % toStore)

        def __split__(corpus, idx_tr, idx_ts):
            corpus_list = []

            for idx in [idx_tr, idx_ts]:
                C = 0
                patients = []

                pbar = tqdm(enumerate(idx))
                for current_idx, pat_idx, in pbar:
                    patient = corpus[pat_idx][0]
                    pbar.set_description("Processing patient %s (index: %s)"
                                         % (patient.index, patient.patient_id))

                    # index += 1
                    patient.patient_id = current_idx
                    C += patient.Cj
                    patients.append(patient)

                corpus_list.append(Corpus(patients, corpus.T, corpus.W, C))

            return tuple(corpus_list)

        skf = StratifiedKFold(n_splits=cv)
        X = list(range(len(corpus)))
        y = corpus.labels

        for fold, (idx_tr, idx_ts) in enumerate(skf.split(X, y)):
            corpus_tr, corpus_ts = __split__(corpus, idx_tr, idx_ts)
            # store data
            __store_data__(os.path.join(toStore, "cv%d" % (fold + 1), 'train'), corpus_tr)
            __store_data__(os.path.join(toStore, "cv%d" % (fold + 1), 'test'), corpus_ts)

            logger.info("Fold (%d) - Training size: %s\nTesting size: %s\n"
                        % (fold + 1, len(corpus_tr), len(corpus_ts)))

    @staticmethod
    def select_only_response(corpus, toStore):

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)

            corpus_file = os.path.join(toStore, "corpus.pkl")

            logger.info("Saving: \n\t%s" % (corpus_file))

            pickle.dump(corpus, open(corpus_file, "wb"))

            logger.info("Data stored in %s" % toStore)

        index = -1
        patients = []
        C = 0

        pbar = tqdm(corpus)

        for p, _ in pbar:
            if not p.isMissingLabel:
                index += 1

                patients.append(p)
                p.patient_id = index
                C += p.Cj
        corpus_filtered = Corpus(patients, corpus.T, corpus.W, C)

        logger.info("Selected %s patients." % len(patients))
        __store_data__(toStore, corpus_filtered)

    @staticmethod
    def read_corpus_from_directory(path, read_metadata=False):
        '''
        Reads data persisted
        :param path: folder containing corpus and metadata files
        :return: Corpus object and metadata (patient ids, data type ids, vocab ids)
        '''
        corpus_file = os.path.join(path, "corpus.pkl")

        corpus = pickle.load(open(corpus_file, "rb"))

        # precompute all labels
        labels = np.zeros(len(corpus))
        for pat, _ in corpus:
            labels[pat.patient_id] = pat.y
        corpus.labels = labels

        if read_metadata:
            metadata_file = os.path.join(path, "meta.pkl")
            meta = pickle.load(open(metadata_file, "rb"))
            return corpus, meta
        return corpus

    @staticmethod
    def generator_sample_data(batch_size, K, T, W, D, M, a=1, b=1, tau=4):
        c = Corpus.build_corpus_sample(K, T, W, D, M, a, b, tau)
        return Corpus.generator(c, batch_size)

    @staticmethod
    def generator(corpus, batch_size):
        generator = DataLoader(corpus, batch_size=batch_size, shuffle=True, collate_fn=Corpus.__collate_mixehr__)
        return generator

    @staticmethod
    def generator_full_batch(corpus):
        generator = DataLoader(corpus, batch_size=len(corpus), shuffle=True, collate_fn=Corpus.__collate_mixehr__)
        return generator

    class Patient(object):
        def __init__(self, patient_id, index, y: int, isMissingLabel: bool = False,
                     words_mapping: Mapping[Set[int], int] = None, Cj: int = 0):
            '''
            Create a new patient.
            :param words_mapping:
            '''
            # key: (data type, word_type)
            # value: frequency
            self.words_dict = words_mapping if words_mapping is not None else {}
            # TODO: change patient_id and index.
            self.patient_id = patient_id
            self.index = index
            # Cj is the lenght of document, which is the sum of each word frequency
            self.Cj = Cj
            self.y = int(y)
            self.isMissingLabel = isMissingLabel

        def append_record(self, type_id, word_id, freq):
            '''
            Append a record to patient words dict
            :param type_id:
            :param word_id:
            :param freq:
            :return:
            '''
            self.words_dict[(type_id, word_id)] = freq
            self.Cj += freq

        def __repr__(self):
            return "<Patient object (%s)>" % self.__str__()

        def __str__(self):
            return "Index (%s). Patient id: %s, Words %s, Cj %s Label %s" % (
                self.patient_id, self.index, len(self.words_dict), self.Cj, self.y)


def run(args):
    cmd = args.cmd
    BASE_FOLDER = args.input
    STORE_FOLDER = args.output
    print(STORE_FOLDER)

    if cmd == 'process':
        path = os.path.join(BASE_FOLDER, 'patient_data.txt')
        # meta = os.path.join(BASE_FOLDER, 'mimic_meta.txt')
        labels = os.path.join(BASE_FOLDER, 'patient_label.csv')
        # Corpus.build_from_mixehr_fileformat(path, meta, labels, STORE_FOLDER,
        #                                     ignore_missing_labels=args.ignore_missing, top_n=args.max)
        Corpus.build_from_mixehr_fileformat(path, labels, STORE_FOLDER,
                                            ignore_missing_labels=args.ignore_missing, top_n=args.max)
    elif cmd == 'split':
        testing_rate = args.testing_rate
        c = Corpus.read_corpus_from_directory(BASE_FOLDER)
        Corpus.split_train_test(c, testing_rate, STORE_FOLDER)
    elif cmd == 'filter':
        c = Corpus.read_corpus_from_directory(BASE_FOLDER)
        Corpus.select_only_response(c, STORE_FOLDER)
    elif cmd == 'stratifiedcv':
        c = Corpus.read_corpus_from_directory(BASE_FOLDER)
        Corpus.split_stratified_cv(c, args.n_splits, STORE_FOLDER)


if __name__ == '__main__':
    # run(parser.parse_args(['process', '-im', '-n', '150', './data/', './store/']))
    # run(parser.parse_args(['split', 'store/', 'store/']))
    # run(parser.parse_args(['process', '-im', '-n', '150', './sample_dataset/', './sample_store/']))
    run(parser.parse_args(['split', 'sample_store/', 'sample_store/']))
    # run(parser.parse_args(['stratifiedcv', '-cv', '5', '/Users/cuent/Downloads/processed_new/mv/out',
    #                        '/Users/cuent/Downloads/processed_new/mv/out']))
