import argparse
import logging
from smixehr_cvb0 import Corpus, MixEHR
import numpy as np
import os
import sys

logger = logging.getLogger("SMixEHR")
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Select one command', dest='cmd')

# default arguments
parser.add_argument('num_topics', help='Number of topics')
parser.add_argument('corpus', help='Path to corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./result/')

# parser train
parser_process = subparsers.add_parser('train', help="Train sMixEHR")
parser_process.add_argument("-it", "--max_iter", help="Maximum number of iterations (Default 500)",
                            type=int, default=500)
parser_process.add_argument("-every", "--save_every", help="Store model every X number of iterations (Default 100)",
                            type=int, default=100)

# parser predict
parser_split = subparsers.add_parser('predict', help="Predict sMixEHR")
parser_split.add_argument("-it", "--max_iter", help="Maximum number of iterations (Default 300)", type=int, default=300)
parser_split.add_argument("-m", "--model", help="Path to model", default='./model/')


def run(args):
    cmd = args.cmd
    corpus = Corpus.read_corpus_from_directory(args.corpus)
    mixehr = MixEHR(int(args.num_topics), corpus, args.output)

    if cmd == 'train':
        logger.info('''
            ======= Parameters =======
            mode: \t\ttraining
            file:\t\t%s
            output:\t\t%s
            num topics:\t\t%s
            max iterations:\t%s
            save every:\t\t%s
            ==========================
        ''' % (args.corpus, args.output, args.num_topics, args.max_iter, args.save_every))

        mixehr.inference_svb(max_iter=args.max_iter, save_every=args.save_every)
    elif cmd == 'predict':
        y_test_true = np.array([p[0].y for p in corpus])
        mixehr.y_test = y_test_true

        logger.info('''
            ======= Parameters =======
            mode: \t\tprediction
            file:\t\t%s
            model:\t\t%s
            output:\t\t%s
            num topics:\t\t%s
            max iterations:\t%s
            ==========================
        ''' % (args.corpus, args.output, args.model, args.num_topics, args.max_iter))

        mixehr.load_model(args.model, int(args.num_topics))
        mixehr.predict(corpus, max_iter=args.max_iter)


if __name__ == '__main__':
    run(parser.parse_args(['train', '40', '../store/', '../model/']))
    run(parser.parse_args(['predict', '40', '../store/', '../result/']))


