import os
import subprocess
from tqdm import tqdm
import shutil

from bin.FuncFea import GetRNAfea
from bin.FuncFea import GetPROfea
from bin.utils import GetFasta
from bin.Hexamer import ReadLogScore


def RNA_StructureFeatures(rna_file, out_prefix):
    '''to compute structure features of rnas and proteins'''

    rna_out = os.path.join(out_prefix, "lncRNA_Structure_features")
    if os.path.exists(rna_out):
        os.remove(rna_out)

    features_workdir = os.path.join(out_prefix, "features_workdir")
    if os.path.exists(features_workdir):
        shutil.rmtree(features_workdir)
    os.mkdir(features_workdir)
    #####################
    # lncRNA structure
    rna_file_part = os.path.join(features_workdir, "tmp.rna.file.")
    rna_file_list = os.path.join(features_workdir, "tmp.filelist")

    rnaID, rna_seq = GetFasta(rna_file)
    rna_Seq = []
    for seq in rna_seq:
        seq = seq.replace('U', 'T')
        rna_Seq.append(seq)
    i = 0
    for rnaid, rnaseq in zip(rnaID, rna_Seq):
        f_tmp = open(rna_file_part + str(i), "w")
        i += 1
        f_tmp.write(">" + rnaid + "\n")
        f_tmp.write(rnaseq + "\n")
        f_tmp.close()

    file_list_cmd = "ls " + features_workdir + " |grep tmp.rna.file > " + rna_file_list
    #print file_list_cmd
    subprocess.call(file_list_cmd, shell=True)

    RNAScore2 = "./data/tools/RNAScore2"

    with open(rna_file_list, "r") as fr:
        bar = tqdm(fr.readlines())
        for tmp in bar:
            tmp = tmp.strip()
            tmpfile = os.path.join(features_workdir, tmp)
            tmpout = os.path.join(features_workdir, tmp + ".r_score")
            rna_cmd = RNAScore2 + " -i " + tmpfile + " -o " + tmpout + " -l 250 -r"
            #print rna_cmd
            subprocess.call(rna_cmd, shell=True)

            combine_cmd = "cat " + tmpout + " >> " + rna_out
            #print combine_cmd
            subprocess.call(combine_cmd, shell=True)
            os.remove(tmpfile)
            os.remove(tmpout)
            bar.set_description("Extract lncRNA Struct Features:")
        fr.close()
    #####################

    os.remove(rna_file_list)
    os.rmdir(features_workdir)
    print("Extract lncRNA struct features has finished.")
    return rna_out


def RNA_SequenceFeatures(rna_file, out_prefix):
    log_hexamer = "./data/tools/Gencode.Refseq.logscore.logscore"
    logscore_dict = ReadLogScore(log_hexamer)

    rna_fea = {}

    rna_ID, rna_seq = GetFasta(rna_file)
    rna_Seq = []
    for seq in rna_seq:
        seq = seq.replace('U', 'T')
        rna_Seq.append(seq)
    length = len(rna_ID)
    for i in tqdm(range(length), desc="Extract lncRNA Sequence Features:"):
        rna_id = rna_ID[i]
        rna_seq = rna_Seq[i]
        nn_edp_fea, rna_lncfea = GetRNAfea(rna_seq, logscore_dict)
        rna_fea[rna_id] = nn_edp_fea+'\t'+rna_lncfea

    RNA_Seq_fea_file = os.path.join(out_prefix, "lncRNA_Sequence_features")
    f = open(RNA_Seq_fea_file, "w")
    for k, v in rna_fea.items():
        f.write(k+" "+v+"\n")
    f.close()
    print("Extract lncRNA sequence features has finished.")
    return RNA_Seq_fea_file


def Protein_StructureFeatures(protein_file, out_prefix):
    '''to compute structure features of rnas and proteins'''

    protein_out = os.path.join(out_prefix, "protein_Structure_features")
    if os.path.exists(protein_out):
        os.remove(protein_out)

    #####################
    # protein structure
    features_workdir = os.path.join(out_prefix, "features_workdir")
    if os.path.exists(features_workdir):
        shutil.rmtree(features_workdir)
    os.mkdir(features_workdir)
    protein_file_part = os.path.join(features_workdir, "tmp.protein.file.")
    protein_file_list = os.path.join(features_workdir, "tmp.filelist")

    stride_dat = "./data/tools/stride.dat"
    stride_cmd = "cp " + stride_dat + " " + os.path.abspath('.')
    tmp_stride_dat = os.path.join(os.path.abspath('.'), "stride.dat")
    subprocess.call(stride_cmd, shell=True)

    proID, proSeq = GetFasta(protein_file)
    i = 0
    for proid, proseq in zip(proID, proSeq):
        f_tmp = open(protein_file_part + str(i), "w")
        i += 1
        f_tmp.write(">" + proid + "\n")
        f_tmp.write(proseq + "\n")
        f_tmp.close()

    file_list_cmd = "ls " + features_workdir + \
        " |grep tmp.protein.file > " + protein_file_list
    #print file_list_cmd
    subprocess.call(file_list_cmd, shell=True)

    RNAScore2 = "./data/tools/RNAScore2"

    with open(protein_file_list, "r") as fp:
        bar = tqdm(fp.readlines())
        for tmp in bar:
            tmp = tmp.strip()
            tmpfile = os.path.join(features_workdir, tmp)
            tmpout = os.path.join(features_workdir, tmp + ".pro_score")
            protein_cmd = RNAScore2 + " -i " + tmpfile + " -o " + tmpout + " -p"
            #print protein_cmd
            subprocess.call(protein_cmd, shell=True)

            combine_cmd = "cat " + tmpout + " >> " + protein_out
            #print combine_cmd
            subprocess.call(combine_cmd, shell=True)
            os.remove(tmpfile)
            os.remove(tmpout)
            bar.set_description("Extract protein Struct Features:")

        fp.close()
    #####################

    os.remove(protein_file_list)
    os.remove(tmp_stride_dat)
    os.rmdir(features_workdir)
    print("Extract protein struct features has finished.")

    return protein_out


def Protein_SequenceFeatures(pro_file, out_prefix):
    '''generate rna, protein features'''
    pro_fea = {}

    pro_ID, pro_Seq = GetFasta(pro_file)

    length = len(pro_ID)
    for i in tqdm(range(length), desc="Extract protein Sequence Features:"):
        pro_id = pro_ID[i]
        pro_seq = pro_Seq[i]
        aa_edp_fea = GetPROfea(pro_seq)
        pro_fea[pro_id] = aa_edp_fea

    pro_Seq_fea_file = os.path.join(out_prefix, "protein_Sequence_features")
    f = open(pro_Seq_fea_file, "w")
    for k, v in pro_fea.items():
        f.write(k+" "+v+"\n")
    f.close()
    print("Extract protein sequence features has finished.")
    return pro_Seq_fea_file


def get_features_files(rna_seq_file, pro_seq_file):
    features_dir = "./data/features_data"
    RNA_SequenceFeatures(rna_seq_file, features_dir)
    RNA_StructureFeatures(rna_seq_file, features_dir)
    Protein_SequenceFeatures(pro_seq_file, features_dir)
    Protein_StructureFeatures(pro_seq_file, features_dir)
    print("Extract features has all finished.")


def read_fea_file(fea_file):
    """read features"""
    fea_dict = {}

    f = open(fea_file, 'r')
    for line in f.readlines():
        line = line.strip()
        if len(line.split()) < 5:
            continue
        fea_dict[line.split()[0]] = [float(x) for x in line.split()[1:]]
    f.close()

    return fea_dict


def read_features_files(pairs_file):
    features_dir = "./data/features_data"
    rna_stu_fea_file = os.path.join(features_dir, "lncRNA_Structure_features")
    pro_stu_fea_file = os.path.join(features_dir, "protein_Structure_features")
    rna_seq_fea_file = os.path.join(features_dir, "lncRNA_Sequence_features")
    pro_seq_fea_file = os.path.join(features_dir, "protein_Sequence_features")
    rna_id = []
    pro_id = []

    f = open(pairs_file, 'r')
    for line in f.readlines():
        line = line.strip()
        rna_id.append(line.split()[0])
        pro_id.append(line.split()[1])
    f.close()

    rna_stru_fea = read_fea_file(rna_stu_fea_file)
    pro_stru_fea = read_fea_file(pro_stu_fea_file)
    rna_seq_fea = read_fea_file(rna_seq_fea_file)
    pro_seq_fea = read_fea_file(pro_seq_fea_file)
    # print(len(rna_seq_fea['n1114']))  # 303 : 256 36 1 2 8
    # print(len(pro_seq_fea['Q15717']))  # 343
    # print(len(rna_stru_fea['n1114']))  # 30
    # print(len(pro_stru_fea['Q15717']))  # 50
    features = []

    for r, p in zip(rna_id, pro_id):
        features.append(rna_seq_fea[r]+pro_seq_fea[p]+rna_stru_fea[r]+pro_stru_fea[p])
    return features
