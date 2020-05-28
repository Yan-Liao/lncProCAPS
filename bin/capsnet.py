import torch
import torch.nn as nn
import torch.nn.functional as func
import math


def squash(x):
    length2 = x.pow(2).sum(dim=2)+1e-7
    length = length2.sqrt()
    x = x*(length2/(length2+1)/length).view(x.size(0), x.size(1), -1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations=3):
        super().__init__()
        self.n_iterations = n_iterations
        self.b = torch.zeros((input_caps, output_caps))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        if u_predict.is_cuda:
            self.b = self.b.cuda()
        self.b.zero_()
        c = func.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))  # 这块带上batch是因为每个样本的c都不一样
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = func.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim):
        super().__init__()
        self.input_caps = input_caps
        self.input_dim = input_dim
        self.output_caps = output_caps
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.Tensor(self.input_caps, self.input_dim, self.output_caps * self.output_dim))
        self.routing_module = AgreementRouting(self.input_caps, self.output_caps)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    # 输入的x格式为：(batch_m, input_caps, input_dim)
    def forward(self, u):
        u = u.unsqueeze(2)
        u_predict = u.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        probs = v.pow(2).sum(dim=2).sqrt()
        return v, probs


# loss_fun
class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)  # 按行传播 相当于one-hot
        losses = t.float() * func.relu(self.m_pos - lengths).pow(2) + \
            self.lambda_ * (1. - t.float()) * func.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


class NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna_kmer1 = nn.Linear(256, 50)
        self.rna_kmer2 = nn.Linear(50, 30)

        self.pro_kmer1 = nn.Linear(343, 60)
        self.pro_kmer2 = nn.Linear(60, 50)

        self.seq_togather1 = nn.Linear(127, 100)
        self.seq_togather2 = nn.Linear(100, 60)
        self.seq_togather3 = nn.Linear(60, 30)

        self.stru_togather1 = nn.Linear(80, 80)
        self.stru_togather2 = nn.Linear(80, 80)
        self.stru_togather3 = nn.Linear(80, 40)
        self.stru_togather4 = nn.Linear(40, 30)

        self.caps = CapsLayer(input_caps=2, input_dim=30, output_caps=1, output_dim=30)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        rna_kmer = x[:, :256]
        rna_other_seq = x[:, 256:303]
        pro_kmer = x[:, 303:646]
        rna_stru = x[:, 646:676]
        pro_stru = x[:, 676:726]

        rna_kmer = self.prelu(self.dropout(self.rna_kmer1(rna_kmer)))
        rna_kmer = torch.tanh(self.rna_kmer2(rna_kmer))

        pro_kmer = self.prelu(self.dropout(self.pro_kmer1(pro_kmer)))
        pro_kmer = torch.tanh(self.pro_kmer2(pro_kmer))

        seq_togater = torch.cat((rna_kmer, pro_kmer, rna_other_seq), dim=1)
        seq_togater = self.prelu(self.dropout(self.seq_togather1(seq_togater)))
        seq_togater = self.prelu(self.dropout(self.seq_togather2(seq_togater)))
        seq_togater = torch.tanh(self.seq_togather3(seq_togater))

        stru_togater = torch.cat((rna_stru, pro_stru), dim=1)
        stru_togater = self.prelu(self.dropout(self.stru_togather1(stru_togater)))
        stru_togater = self.prelu(self.dropout(self.stru_togather2(stru_togater)))
        stru_togater = self.prelu(self.dropout(self.stru_togather3(stru_togater)))
        stru_togater = torch.tanh(self.stru_togather4(stru_togater))

        seq_togater = seq_togater.view(seq_togater.shape[0], 1, 30)
        stru_togater = stru_togater.view(stru_togater.shape[0], 1, 30)
        togater = torch.cat((seq_togater, stru_togater), dim=1)

        v, probs = self.caps(togater)

        return probs.squeeze(1)


# # 二分类版
# class MarginLoss(nn.Module):
#     def __init__(self, m_pos, m_neg, lambda_):
#         super(MarginLoss, self).__init__()
#         self.m_pos = m_pos
#         self.m_neg = m_neg
#         self.lambda_ = lambda_
#
#     def forward(self, lengths, targets, size_average=True):
#         losses = targets * func.relu(self.m_pos - lengths).pow(2) + \
#             self.lambda_ * (1 - targets) * func.relu(lengths - self.m_neg).pow(2)
#         return losses.mean() if size_average else losses.sum()