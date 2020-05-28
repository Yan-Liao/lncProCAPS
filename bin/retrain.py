from tqdm import tqdm
import torch
import torch.utils.data as dataset
import torch.nn as nn

from bin.get_features import read_features_files
from bin.utils import normalize
from bin.capsnet import NET


def train(train_data, train_label, epoch, batch_size, lr, cuda, model_save_file):
    torch_dataset = dataset.TensorDataset(train_data, train_label)
    loader = dataset.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # get net,optimizer and loss_fun
    net = NET()
    if cuda:
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # train
    for e in tqdm(range(epoch), desc="Retraining:"):

        for batch_idx, (train_data, train_label) in enumerate(loader):
            if cuda:
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            train_probs = net(train_data)
            train_loss = loss_fn(train_probs, train_label)
            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1e+7)
            optimizer.step()

    torch.save(net.state_dict(), model_save_file)


def retrain(positive_pairs_file, negative_pairs_file, epoch, batch_size, lr, cuda):

    p_train_data = read_features_files(positive_pairs_file)
    n_train_data = read_features_files(negative_pairs_file)

    train_data = p_train_data + n_train_data

    train_label = torch.cat((torch.ones(len(p_train_data)), torch.zeros(len(n_train_data))))

    mean_save_file = "./data/model/user_defined/user_defined_normalize_mean"
    std_save_file = "./data/model/user_defined/user_defined_normalize_std"
    train_data = normalize(train_data, mean_save_file, std_save_file)  # normalize data, and save mean and std in a file

    train_data = torch.tensor(train_data)

    model_save_file = "./data/model/user_defined/user_defined_parameter.pkl"
    train(train_data, train_label, epoch, batch_size, lr, cuda, model_save_file)

    print("Retrain Finshed.")
