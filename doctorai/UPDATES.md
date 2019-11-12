Files I added on top of original source codes:
* `process_mimic2.py`
* `show_n_input_output.py`

Steps to reproduce:

1. Comment out line 61, ncomment line 60, run `python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv full_icd9`
2. Comment out line 60, ncomment line 61, run `python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv 3digit_icd9`
3. run `python process_mimic2.py`
4. to train: `python doctorAI.py visit 4894 label 942 trained_model`
5. to test: `python testDoctorAI.py trained_model.9.npz visit.test label.test "[200, 200]"`

Results:
`recall@10:0.300923, recall@20:0.440109, recall@30:0.529216`

Total number of patients: 7537