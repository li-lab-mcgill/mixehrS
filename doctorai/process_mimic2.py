"""
Custom file to preprocess output of process_mimic.py so that it can be input to the model.

NOTE:
The label dataset (let us call this "label file") needs to have the same format as the "visit file". The important thing is, time steps of both "label file" and "visit file" need to match. DO NOT train Doctor AI with labels that is one time step ahead of the visits. It is tempting since Doctor AI predicts the labels of the next visit. But it is internally taken care of. 
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


# Load all the files
full_icd9_seqs = load_file('full_icd9.seqs')
full_icd9_dates = load_file('full_icd9.dates')
full_icd9_pids = load_file('full_icd9.pids')
full_icd9_types = load_file('full_icd9.types')

small_icd9_seqs = load_file('3digit_icd9.seqs')
small_icd9_dates = load_file('3digit_icd9.dates')
small_icd9_pids = load_file('3digit_icd9.pids')
small_icd9_types = load_file('3digit_icd9.types')


# Generating train-val-test indices
train_ratio = 0.8
val_ratio = 0.1

indices = list(range(len(full_icd9_seqs)))
random.seed(2019)
random.shuffle(indices)

train_split =  int(train_ratio * len(full_icd9_seqs))
val_split = int(val_ratio * len(full_icd9_seqs))
train_idx = indices[:train_split]
val_idx = indices[train_split:train_split+val_split]
test_idx = indices[train_split+val_split:]

# Split train-val-test
X_train = [full_icd9_seqs[i] for i in train_idx]
X_val = [full_icd9_seqs[i] for i in val_idx]
X_test = [full_icd9_seqs[i] for i in test_idx]

y_train = [small_icd9_seqs[i] for i in train_idx]
y_val = [small_icd9_seqs[i] for i in val_idx]
y_test = [small_icd9_seqs[i] for i in test_idx]

print "Total number of patients:", len(full_icd9_seqs)
print "Number of training patients:", len(X_train), len(y_train)
print "Number of validation patients:", len(X_val), len(y_val)
print "Number of test patients:", len(X_test), len(y_test)


# Save
save_file(X_train, "visit.train")
save_file(X_val, "visit.valid")
save_file(X_test, "visit.test")
save_file(y_train, "label.train")
save_file(y_val, "label.valid")
save_file(y_test, "label.test")