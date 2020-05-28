import torch
import numpy as np

from bin.capsnet import NET
from bin.get_features import read_features_files
from bin.utils import read_label_file
from bin.utils import value


def predict(predict_pairs, output_file, cuda, retrain, label_file):
    if retrain:
        parameters_file = "./data/model/user_defined/user_defined_parameter.pkl"
        normalize_mean = np.load("./data/model/user_defined/user_defined_normalize_mean.npy")
        normalize_std = np.load("./data/model/user_defined/user_defined_normalize_std.npy")
    else:
        parameters_file = "./data/model/default/default_parameter.pkl"
        normalize_mean = np.load("./data/model/default/default_normalize_mean.npy")
        normalize_std = np.load("./data/model/default/default_normalize_std.npy")

    net = NET()
    net.load_state_dict(torch.load(parameters_file))

    predict_data = read_features_files(predict_pairs)
    predict_data_np = np.array(predict_data, dtype=float)
    predict_data_np = (predict_data_np - normalize_mean) / normalize_std  # normalize
    predict_data = predict_data_np.tolist()
    predict_data = torch.tensor(predict_data)

    if label_file != None:
        label = read_label_file(label_file)
        label = torch.tensor(label)

    if cuda:
        net = net.cuda()
        predict_data = predict_data.cuda()

    net.eval()
    with torch.no_grad():
        predict_probs = net(predict_data)
        one = torch.ones_like(predict_probs)
        zero = torch.zeros_like(predict_probs)
        predict_pred = torch.where(predict_probs > 0.5, one, zero)
        if label_file != None:
            predict_accuracy, predict_sn, predict_sp, predict_f1 = value(predict_pred, label)
            print('\npredict_accuracy:', predict_accuracy,
                '\tpredict_sn:', predict_sn,
                '\tpredict_sp:', predict_sp,
                '\tpredict_F1:', predict_f1)

    rna_id = []
    pro_id = []
    f = open(predict_pairs, 'r')
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        rna_id.append(line.split()[0])
        pro_id.append(line.split()[1])
    f.close()

    predict_pred = predict_pred.type(torch.int).cpu().numpy()
    f = open(output_file, 'w')
    if label_file != None:
        label = label.type(torch.int).numpy()
        f.write('predict_accuracy:'+str(predict_accuracy) +
                '\tpredict_sn:'+str(predict_sn) +
                '\tpredict_sp:'+str(predict_sp) +
                '\tpredict_F1:'+str(predict_f1) +
                "\nrna_id  pro_id  label  predict('1' is interact,'0' is not interact)\n")
        for idx in range(len(rna_id)):
            f.write(rna_id[idx] + ' ' + pro_id[idx] + ' ' + str(label[idx]) + ' ' + str(predict_pred[idx]) + '\n')
    else:

        f.write("rna_id  pro_id  predict('1' is interact,'0' is not interact)\n")
        for idx in range(len(rna_id)):
            f.write(rna_id[idx] + '  ' + pro_id[idx] + '  ' + str(predict_pred[idx]) + '\n')
    f.close()
    print("\nPrediction is finished, check the result at:"+output_file)
