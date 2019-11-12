Doctor AI
=========================================

Doctor AI is a automatic diagnosis machine that predicts medical codes that occur in the next visit, while also predicting the time duration until the next visit.

#### Relevant Publications

Doctor AI implements an algorithm introduced in the following:

	Doctor AI: Predicting Clinical Events via Recurrent Neural Networks  
	Edward Choi, Mohammad Taha Bahadori, Andy Schuetz, Walter F. Stewart, Jimeng Sun  
	arXiv preprint arXiv:1511.05942
	
	Medical Concept Representation Learning from Electronic Health Records and its Application on Heart Failure Prediction  
	Edward Choi, Andy Schuetz, Walter F. Stewart, Jimeng Sun  
	arXiv preprint arXiv:1602.03686

#### Running Doctor AI

**STEP 1: Installation**  

1. Install [python](https://www.python.org/), [Theano](http://deeplearning.net/software/theano/index.html). We use Python 2.7, Theano 0.7. Theano can be easily installed in Ubuntu as suggested [here](http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu)

2. If you plan to use GPU computation, install [CUDA](https://developer.nvidia.com/cuda-downloads)

3. Download/clone the Doctor AI code  

**STEP 2: Preparing training data**  

0. You can use "process_mimic.py" to process MIMIC-III dataset and generate a suitable training dataset for Doctor AI. Place the script to the same location where the MIMIC-III CSV files are located, and run the script. Instructions are described inside the script. However, I recommend the readers to read the following steps to understand the structure of the training data and learn how to prepare their own dataset.

1. Doctor AI's training dataset needs to be a Python Pickled list of list of list. Each list corresponds to patients, visits, and medical codes (e.g. diagnosis codes, medication codes, procedure codes, etc.)
First, medical codes need to be converted to an integer. Then a single visit can be seen as a list of integers. Then a patient can be seen as a list of visits.
For example, [5,8,15] means the patient was assigned with code 5, 8, and 15 at a certain visit.
If a patient made two visits [1,2,3] and [4,5,6,7], it can be converted to a list of list [[1,2,3], [4,5,6,7]].
Multiple patients can be represented as [[[1,2,3], [4,5,6,7]], [[2,4], [8,3,1], [3]]], which means there are two patients where the first patient made two visits and the second patient made three visits.
This list of list of list needs to be pickled using cPickle. We will refer to this file as the "visit file".

2. The total number of unique medical codes is required to run Doctor AI.
For example, if the dataset is using 14,000 diagnosis codes and 11,000 procedure codes, the total number is 25,000. 

3. The label dataset (let us call this "label file") needs to have the same format as the "visit file".
The important thing is, time steps of both "label file" and "visit file" need to match. DO NOT train Doctor AI with labels that is one time step ahead of the visits. It is tempting since Doctor AI predicts the labels of the next visit. But it is internally taken care of.
You can use the "visit file" as the "label file" if you want Doctor AI to predict the exact codes. 
Or you can use a grouped codes as the "label file" if you are okay with reasonable predictions and want to save time. 
For example, ICD9 diagnosis codes can be grouped into 283 categories by using [CCS](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) groupers. 
We STRONGLY recommend you do this, because the number of medical codes can be as high as tens of thousands, 
which can cause not only low predictive performance but also memory issues. (The high-end GPUs typically have only 12GB of VRAM)

4. Same as step 2, you will need to remember the total number of unique codes in the "label file".
If you are using "visit file" as the "label file", than the number of unique codes will be the same, of course.

5. The "visit file" and "label file" need to have 3 sets respectively: training set, validation set, and test set.
The file extension must be ".train", ".valid", and ".test" respectivley.  
For example, if you want to use a file named "my_visit_sequences" as the "visit file", then Doctor AI will try to load "my_visit_sequences.train", "my_visit_sequences.valid", and "my_visit_sequences.test".  
This is also true for the "label file"

5. You can use the time duration between visits as an additional source of information. Let us call this "time file".
"time file" needs to be prepared as a Python Pickled List of List. Each list corresponds to patients and the duration between each visit.
For example, given a "visit file" [[[1,2,3], [4,5,6,7]], [[2,4], [8,3,1], [3]]], its corresponding "time file" should look like [[0, 15], [0, 45, 23]].
Of course, the numbers are fake, but the important thing is that the duration for the first visit needs to be zero. 
Use "--time\_file" option to use "time file"
Remember that the ".train", ".valid", ".test" rule also applies to the "time file" as well.

**Additional: Predicting time duration until next visit**  
In addtion to predicting the codes of the next visit, you can make Doctor AI predict the time duration until next visit. 
Use "--predict\_time" option to do this. And obviously, predicting time requires the "time file".  
Time prediction also comes with many hyperparameters such as "--tradeoff", "--L2\_time", "--use\_log\_time". 
Refer to "--help" for more detailed information

**Additional: Using your own medical code representations**  
Doctor AI internally learns vector representation of medical codes while training. These vectors are initialized with random values of course.  
You can, however, also provide medical code representations, if you have one. (They can be easily trained by using Skip-gram like algorithms.)
If you want to provide the medical code representations, it has to be a list of list (basically a matrix) of N rows and M columns where N is the number of unique codes in your "visit file" and M is the size of the code representations.
Specify the path to your code representation file using "--embed\_file".  
For more details regarding the training of medical code representations and using them for predictive tasks, please refer to the second paper of the "Related Publication" section.  
Additionally even if you provided your own medical code representations, you can re-train (a.k.a fine-tune) them as you train Doctor AI. 
Use "--embed\_finetune" option to do this. If you are not providing your own medical code representations, Doctor AI will use randomly initialized one, which obviously requires this fine-tuning process. Since the default is to use the fine-tuning, you do not need to worry about this.

**STEP 3: Running Doctor AI**  

1. The minimum input you need to run Doctor AI is the "visit file", the number of unique medical codes in the "visit file", 
the "label file", the number of unique medical codes in the "label file", and the output path. The output path is where the learned weights will be saved.  
`python doctorAI.py <visit file> <# codes in the visit file> <label file> <# codes in the label file> <output path>`  

2. Specifying `--verbose` option will print training process after each 10 mini-batches.

3. You can specify how many GRU layers you want to use by using "--hidden\_dim\_size" option.
For example "--hidden\_dim\_size \[400,200\]" will give you a two layer GRU where the lower layer uses a 400-dimensional hidden layer 
and the upper layer uses a 200-dimensional hidden layer.

4. Additional options can be specified such as the size of the embedding layer, batch size, the number of epochs, dropout rate, etc. Detailed information can be accessed by `python doctorAI.py --help`

**STEP 4: Getting your results**  

Doctor AI checks the validation cross entropy after each epoch, and if it is lower than all previous value, it will save the current model. The model file is generated by [numpy.savez_compressed](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.savez_compressed.html).

**Step 5: Testing your model**

1. Using the file "testDoctorAI.py", you can calculate the recall@10,20,30 for the code prediction and R^2 for the time prediction. First you need to have a trained model that was saved by numpy.savez\_compressed. Note that you need to know the configuration with which you trained Doctor AI (use of "time file", use of "--use\_log\_time", value of "--hidden\_dim\_size", etc.)

2. Again, you need the "visit file" and "label file" prepared in the same way. This time, however, you do not need to follow the ".train", ".valid", ".test" rule. The testing script will try to load the file name as given.

3. Using additional options such as "--hidden\_dim\_size" and "--use\_log\_time", you should use exactly the same configuration with which you trained the model. For more detailed information, use "--help" option.

4. To evaluate the time prediction performance, we provide R^2 error. In order to calculate this, you need to provide the mean value of all durations in the "time file" you used to train Doctor AI. (You must ignore the 0 duration of the first visits or course) Use "--mean\_duration" option to do this.

5. The minimum input to run the testing script is the "model file", "visit file", "label file", and "hidden dim size".  
`python testDoctorAI.py <model file> <visit file> <label file> <hidden_dim_size>`
