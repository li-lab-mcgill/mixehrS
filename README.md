# MixEHR-S

MixEHR-S is a topic modelling framework that jointly infers specialist-disease topics from the EHR data and predict patient-specific labels. 
The specialist diagnosed ICD codes is learned in the training process using the unsupervised topic modelling componnent.
By introducing the heterogeneous specialist topics into the modelling, it can learn high interpretable disease representations that lead to more accurate predictions. 
For the supervised prediction task, MixEHR-S leverages the inferred topic  mixture to predict patient's binary label with Bayesian regression component. 
The binary prediction task could take any form such as disease prediction, drug recommendation, or death prediction. 
We also consider extensions of our current model, such as multi-class classification and temporal prediction.

The proababilistic graphical model of MixEHR-S is shown:


<img src="https://github.com/li-lab-mcgill/mixehrS/blob/master/figures/PGM.jpg" width="500" height="350">



# Relevant Publications

This published code is referenced from following paper:

>Ziyang Song, Xavier Sumba Toral, Yixin Xu, Aihua Liu, Liming Guo, GuidoPowell, Aman Verma, David Buckeridge, Ariane Marelli, and Yue Li. 2021. Supervised 
multi-specialist topic model with applications on large-scaleelectronic health record data. In12th ACM Conference on Bioinformatics,Computational Biology, 
and Health Informatics August 1–4, 2021, Virtual dueto COVID-19.ACM, New York, NY, USA, 25 pages. https://doi.org/10.1145/1122445.11224561.

# Dataset

We evaluated MixEHR-S on 3 different datasets: CHD dataset from large claim database, COPD dataset and diabetes dataset from PopHR database. 
For these datasets, we consider each patients, specialists, ICD-9 codes as documents, data types, and words (vocabulary), respectively. 
The specific (specialist, ICD code) pair are organized already and calculated to obtain frequency.
These information are included in mixehrS/data/data.txt file.

We also need binary label information for each patient to preform prediction. 
In this three datasets, response label are CHD disease label, COPD disease label, the drug usage after 6 months of diabetes diagnosis. 
These information are included in mixehrS/data/label.txt file.

As the large claim database and PopHR database are confidential, we release a toy data in path "mixehrS/data/"

# Code Description

## STEP 1: Process Dataset

The input data file need to be processed into built-in data structure "Corpus". You can use "mixehrS/code/corpus.py" to process dataset and generate a suitable data structure for MixEHR-S.
Place dataset to specific path "mixehrS/data/" or change the corresponding runnning code. You can run following code:

    run(parser.parse_args(['process', '-im', '-n', '150', './data/', './store/']))
    
you also need to split the dataset into train/validation/test subset. The data path and detailed split ratio could be edited:
    
    run(parser.parse_args(['split', 'store/', 'store/']))

## STEP 2: Topic Modelling

After process dataset and obtained trainable data, you can run "mixehrS/code/main.py" to perform unsupervised topic modelling for each dataset. 
Hyperparameters of training stage include number of latent topics,  training epoches, and parameters related to stochastic learning. The defaulted values are placed in the code. 
The first argument should be train. Th topics number, data path and epoches could choose adequate value. The execution code is:

    run(parser.parse_args(['train', '40', '../store/', '../model/']))
    

## STEP 3: Label Prediction

With the saved models stored in training stage, you can used these models to preform patient specific label prediction with code in "mixehrS/code/main.py". 
The number of latent topics should be same with the number of saved model (training stage). 
The test set should be used in label prediction task. The execution code is:

    run(parser.parse_args(['predict', '40', '../store/', '../result/']))
    
##STEP 4: Hyper-parameter Tuning

For MixEHR-S, the topic number should be tuned on the validation set. We chose the number of topics which gives the highest likelihood on the validation set.
After we obtained the optimal topics number, MixEHR-S can evaluated on train set.

##STEP 5: Prepare Your Own Dataset

The required dataset may not be EHR ICD data. Any dataset includes words and diverse data types could be considered. 
For example, ACT code and drug code in EHR data can only be organized into trainable dataset. You 

Your prepared data should have two files: text data and label data.
- data.txt: the document (patient) data with ID, word (ICD code), data type (specialist), frequency.

                            Headers:
                            id, word, data_type, frequency 


- label.txt: for each document ID, whether this document has a corresponding response. In this paper, the response label is disease label and drug useage. It could be death flag and others. 

                            Headers:
                            id,response

                            1 indicates patients have certain response (have disease, use drug after a period, are already dead and so on). 0 is opposite. 





