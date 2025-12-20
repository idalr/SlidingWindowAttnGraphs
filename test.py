import torch

rlx = torch.load('datasets/pts/rl_data_1.pt', weights_only=False)
npf = torch.load('datasets/pts/np_f_data_1.pt', weights_only=False)
npm = torch.load('datasets/pts/np_m_data_1.pt', weights_only=False)

print('Done.')

