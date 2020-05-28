import torch
import argparse

from bin.get_features import get_features_files
from bin.retrain import retrain
from bin.predict import predict

torch.manual_seed(1)
# Hyper parameters
K_FOLD = 6
EPOCH = 60
BATCH_SIZE = 100
LR = 0.001


def main():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]',
                                     description='An lncRNA-protein interaction predict tool based on  neural network and capsule net')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    # step 1: Extracted features
    parser.add_argument('-e_f', '--ext_fea', action='store', dest='extracted_features', choices=[True, False], type=bool,
                        default=False,
                        help='Whether to extract features. Default is "False"')

    parser.add_argument('-r_f', '--rna_fasta', action='store', dest='rna_sequence_file',
                        help='(If --ext_fea is True,required this.)Need rna sequence fasta file, witch contain /'
                             'predict rna sequence. And if you want to retrain the model, this file must contain/'
                             ' retrain rna sequence,too.')

    parser.add_argument('-p_f', '--pro_fasta', action='store', dest='protein_sequence_file',
                        help='(If --ext_fea is True,required this.)Need protein sequence fasta file, witch contain/'
                             ' predict protein sequence. And if you want to retrain the model, this file must contain/'
                             ' retrain protein sequence,too.')

    # step 2:predict or retrain
    parser.add_argument('-m', '--mode', action='store', dest='mode', choices=['predict', 'retrain', None], type=str,
                        default=None,
                        help='(Required) The mode used for predict or retrain. Default is "predict"')

    parser.add_argument('-o', '--out', action='store', dest='output_file',
                        default="./data/output_data/result",
                        help='The output file of result.(File can not exist,but path must exist.)')

    parser.add_argument('-d', '--device', action='store', dest='CUDA', choices=[True, False], type=bool,
                        default=False,
                        help='Whether to use GPU to accelerate calculation.(Default is "False")')
    # 2.1 train
    parser.add_argument('-i_p', '--interact_pairs', action='store', dest='interact_pairs_file',
                        help='(If --mode is "retrain",required this.) Interact_pairs_file.')

    parser.add_argument('-n_p', '--not_interact_pairs', action='store', dest='not_interact_pairs_file',
                        help='(If --mode is "retrain",required this.) NOT_interact_pairs_file.')

    # 2.2 test
    parser.add_argument('-u_r', '--use_retrained_model', action='store', dest='use_retrained_model',
                        choices=[True, False], type=bool, default=False,
                        help='Whether to use retrained model to predict.(Default is "False")')

    parser.add_argument('-p', '--predict_pairs', action='store', dest='predict_pairs_file',
                        help='(If --mode is "predict",required this.) Predict_pairs_file')

    parser.add_argument('-l', '--predict_pairs_label', action='store', dest='predict_pairs_label_file',
                        help='If your predict pairs have labels, input file can get forecast accuracy')

    args = parser.parse_args()
    print(args)

    if args.extracted_features:
        get_features_files(args.rna_sequence_file, args.protein_sequence_file)

    if args.mode == "retrain":
        retrain(args.interact_pairs_file, args.not_interact_pairs_file, EPOCH, BATCH_SIZE, LR, args.CUDA)

    if args.mode == "predict":
        predict(args.predict_pairs_file, args.output_file, args.CUDA, args.use_retrained_model,
                args.predict_pairs_label_file)


if __name__ == '__main__':
    main()

