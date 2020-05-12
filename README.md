# lncProCAPS
An lncRNA-protein interaction predict tool based on  neural network and capsule net.  

Need Linux operating system.  
install:Download the package, and make sure you have installed these package:  
numpy(>=1.16.4)  
torch  
argparse  
tqdm(>=4.32.1)  
shutil(>=1.0.0)  

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

optional arguments:  
  -h, --help            show this help message and exit 
    
  -v, --version         show program's version number and exit 
    
  -e_f, --ext_fea {True,False}  
                        Whether to extract features. Default is "False"  
                          
  -r_f, --rna_fasta (RNA_seq_file's path)  
                        (If --ext_fea is True,required this.)Need rna sequence
                        fasta file, witch contain /predict rna sequence. And
                        if you want to retrain the model, this file must
                        contain/ retrain rna sequence,too.  
                          
  -p_f, --pro_fasta (protein_seq_file's path)  
                        (If --ext_fea is True,required this.)Need protein
                        sequence fasta file, witch contain/ predict protein
                        sequence. And if you want to retrain the model, this
                        file must contain/ retrain protein sequence,too.  
                          
  -m, --mode {predict,retrain,None}  
                        (Required) The mode used for predict or retrain.  
                        Default is "None".  
                        (When you are extracting features, this optinon shoule be None.)     
                          
  -o, --out (output_file's path)  
                        The output file of result.(File can not exist,but path
                        must exist.)  
			default="./data/output_data/result"  
        
  -d, --device {True,False}  
                        Whether to use GPU to accelerate calculation.(Default
                        is "False")  
                          
  -i_p, --interact_pairs (interact_pairs_file's path)  
                        (If --mode is "retrain",required this.)  
                        Interact_pairs_file.  
                          
  -n_p, --not_interact_pairs (not interact_pairs_file's path)  
                        (If --mode is "retrain",required this.)  
                        NOT_interact_pairs_file.  
                          
  -u_r, --use_retrained_model {True,False}  
                        Whether to use retrained model to predict.(Default is
                        "False")  
                          
  -p, --predict_pairs (predict_pairs_file's path)  
                        (If --mode is "predict",required this.)  
                        Predict_pairs_file  
                          
  -l, --predict_pairs_label (predict_pairs_label_file's path)  
                        If your predict pairs have labels, input file can get
                        forecast accuracy.  
