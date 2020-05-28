import numpy as np


def GetFasta(inputfile):
    '''Get sequence from input'''

    f = open(inputfile, 'r')    # input file
    tmpseq = ''
    seqlist = []
    seqID = []

    for line in f.readlines():
        line = line.strip()
        if not len(line):
            continue
        elif line[0] == '>':
            seqID.append(line.split()[0][1:])
            if tmpseq != '':
                seqlist.append(tmpseq)
            tmpseq = ''
        else:
            tmpseq += line.upper()
    seqlist.append(tmpseq)      ## append the last sequence
    f.close()

    return [seqID, seqlist]


def read_label_file(label_file):
    label = []
    f = open(label_file, 'r')
    for line in f.readlines():
        line = line.strip()
        assert len(line) <= 1
        if len(line) == 0:
            continue
        else:
            label.append(int(line))
    return label


def normalize(train_data, mean_save_file, std_save_file):
    """对数据进行标准化化操作（输入为list，输出也为list）"""
    train_data_np = np.array(train_data, dtype=float)

    mean = np.mean(train_data_np, axis=0, keepdims=True)
    std = np.std(train_data_np, axis=0, ddof=1, keepdims=True)
    index = np.where(std == 0)  # 防止除数为零
    std[index] = 1e-7
    train_data_np = (train_data_np - mean) / std
    train_data = train_data_np.tolist()

    np.save(mean_save_file, mean)
    np.save(std_save_file, std)
    return train_data


def value(pred, label):
    """计算准确率，召回率，精确率，F分数"""
    label = label.cpu().data.numpy()
    pred = pred.cpu().data.numpy()

    accuracy = sum(pred == label) / len(label)
    # 计算Sn，Sp以及F1(对应召回率，精准率以及F1score)
    actually_positive_index = np.argwhere(label == 1)
    actually_positive_pred = pred[actually_positive_index]
    sn = int(sum(actually_positive_pred)) / len(actually_positive_pred)

    pred_positive_index = np.argwhere(pred == 1)
    pred_positive_actually = label[pred_positive_index]
    if len(pred_positive_actually) == 0:
        sp = 0
    else:
        sp = int(sum(pred_positive_actually)) / len(pred_positive_actually)

    if sn+sp == 0:
        f1 = 0
    else:
        f1 = 2 * sn * sp / (sn + sp)

    return accuracy, sn, sp, f1


def Codon2AA2(codon):
    '''convert codon to aa'''
    if codon == "TTT" or codon == "TTC":
        return 'F'
    elif codon == 'TTA' or codon == 'TTG' or codon == 'CTT' or codon == 'CTA' or codon == 'CTC' or codon == 'CTG':
        return 'L'
    elif codon == 'ATT' or codon == 'ATC' or codon == 'ATA':
        return 'I'
    elif codon == 'ATG':
        return 'M'
    elif codon == 'GTA' or codon == 'GTC' or codon == 'GTG' or codon == 'GTT':
        return 'V'
    elif codon == 'GAT' or codon == 'GAC':
        return 'D'
    elif codon == 'GAA' or codon == 'GAG':
        return 'E'
    elif codon == 'TCA' or codon == 'TCC' or codon == 'TCG' or codon == 'TCT':
        return 'S'
    elif codon == 'CCA' or codon == 'CCC' or codon == 'CCG' or codon == 'CCT':
        return 'P'
    elif codon == 'ACA' or codon == 'ACG' or codon == 'ACT' or codon == 'ACC':
        return 'T'
    elif codon == 'GCA' or codon == 'GCC' or codon == 'GCG' or codon == 'GCT':
        return 'A'
    elif codon == 'TAT' or codon == 'TAC':
        return 'Y'
    elif codon == 'CAT' or codon == 'CAC':
        return 'H'
    elif codon == 'CAA' or codon == 'CAG':
        return 'Q'
    elif codon == 'AAT' or codon == 'AAC':
        return 'N'
    elif codon == 'AAA' or codon == 'AAG':
        return 'K'
    elif codon == 'TGT' or codon == 'TGC':
        return 'C'
    elif codon == 'TGG':
        return 'W'
    elif codon == 'CGA' or codon == 'CGC' or codon == 'CGG' or codon == 'CGT':
        return 'R'
    elif codon == 'AGT' or codon == 'AGC':
        return 'S'
    elif codon == 'AGA' or codon == 'AGG':
        return 'R'
    elif codon == 'GGA' or codon == 'GGC' or codon == 'GGG' or codon == 'GGT':
        return 'G'
    # stop codon
    elif codon == 'TAA' or codon == 'TAG' or codon == 'TGA':
        return 'J'
    else:
        return 'Z'     ## IUPAC Ambiguity Codes


