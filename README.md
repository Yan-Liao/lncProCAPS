# lncProCAPS
An lncRNA-protein interaction predict tool based on  neural network and capsule net.

install:Download the package, and make sure you have installed these package:
numpy
torch
argparse
tqdm
shutil

How to use this?
***step 1: extract features
**Prepare: 1.lncRNA faste files; 2.protein faste files.
The files must contain all predict data's ID and sequence. And if you want to retrain the model, it also must contain the retrain data's ID and sequence.(The sample is data/input_data/sample/RNA_seq_file and protein_seq_file)

Then converts the current path in lncProCAPS folder and execute the following command:
*****************************
python lncProCAPS -e_f True -r_f (RNA_seq_file's path) -p_f (protein_seq_file's path)
*****************************
And then wait for extracting features finished.(If your sequence in too long or your data is too much, it may be warning.But dont worry, the process will be fine.)


***step 2:retrain(optional)
If you have the retrain data(many lncRNA-protein interacting pairs and not interacting pairs),you can retrain the model with your own data, and it can make the predict more convincible.
**Prepare:1.lncRNA-protein interacting pairs file; 2.lncRNA-protein not interacting pairs file.
(The sample is data/input_data/sample/retrain_interacting_pairs and retrain_not_interacting_pairs)

Then converts the current path in lncProCAPS folder and execute the following command:
*****************************
python lncProCAPS.py -m retrain -i_p (lncRNA-protein interacting pairs file'spath) -n_p (lncRNA-protein not interacting pairs file'spath) -d False
*****************************
note:-d is an optional choise, if your computer have a NVIDA GPU and your pytorch support CUDA.you can use -d True to accelerate calculation.


***step 3:predict
Now you can predict your data with the default model or retrained model(If you have retrained model following step 2.)
**Prepare:1.predict pairs file; 2.(optional)predict pairs'label file.(The sample is data/input_data/sample/predict_pairs and predict_labels)

Converts the current path in lncProCAPS folder.Then you can predict your data as following command:
*****************************
python lncProCAPS.py -m predict -o (result file path,default="./data/output_data/result") -p (predict pairs file path)
*****************************

If you have retrained model following step 2, and want to predict your data with the retrained model, add this commend: -u_r True
*****************************
python lncProCAPS.py -m predict -o (result file path,default="./data/output_data/result") -p (predict pairs file path) -u_r True
*****************************

If you have predict pairs'label, and want to know the model's accuracy, add this commend: -l (predict_pairs_label_file's path)
*****************************
python lncProCAPS.py -m predict -o (result file path,default="./data/output_data/result") -p (predict pairs file path) -l (predict_pairs_label_file's path)
*****************************

You can see more optional arguments with  
*****************************
python lncProCAPS.py -h
*****************************
