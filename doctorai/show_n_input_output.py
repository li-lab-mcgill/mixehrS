"""
Custom file to print the number of input (visit) and output (label) codes
"""
import cPickle as pickle
import random

def load_file(file):
    with open(file, 'rb') as f:
        loaded = pickle.load(f)
       
    return loaded

def save_file(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

small_icd9_types = load_file('3digit_icd9.types')
full_icd9_types = load_file('full_icd9.types')

print "n_input:", len(full_icd9_types)
print "n_output:", len(small_icd9_types)