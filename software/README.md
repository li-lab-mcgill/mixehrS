# sMixEHR


# Install
TODO

# Dataset

- Patients records: 46450
- Data type: 6
- Word token (vocabulary): 69595
- Record with maximum number of words: 60951

## Process MixEHR dataset
We require the following files containing information about patients.

- mimic_data1.txt: data records for patients, do not contain headers and columns are separated by spaces. I contains the columns SUBJECT_ID (i.e., patient ID), data type ID, variable ID under the data type, frequency  
- mimic_meta.txt: summary of the total number of variable under each of the data types. Do not contain header and columns are separated by a space. Columns: SUBJECT_ID, total
- death_label.txt: for each SUBJECT_ID, whether the patient died at the last admission

0 indicates patients are still alive in the last admission

1 indicates patients are dead in the last admission but obviously alive at early admission

Headers:
  SUBJECT_ID,HOSPITAL_EXPIRE_FLAG

  Separation: space

  Note: _NA_ indicates the patient only has one admission (non-applicable).
  The supervision module will ignore these patients (i.e., not taking into account the
                            corresponding predictive likelihood). It treats as missing data; missing
                            data is represented with -1.

Generate a MixEHR corpus by executing the following command:
```bash
python corpus.py path output
```

path: directory containing MixEHR data.

output: directory to store processed data.
