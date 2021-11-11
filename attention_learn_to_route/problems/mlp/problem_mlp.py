from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.mlp.state_mlp import StateMLP
from utils.beam_search import beam_search

class MLP(object):

    NAME = 'mlp'  # Split Delivery Vehicle Routing Problem


    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['loc'].size()[0], dataset['loc'].size()[-2]

        a_prev = None

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        pi_with_depot = torch.cat((torch.zeros(batch_size, 1, dtype=torch.uint8, device=pi.device), pi), 1)
        d = loc_with_depot.gather(1, pi_with_depot[..., None].expand(*pi_with_depot.size(), loc_with_depot.size(-1)))

        d_norm = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2)
        latency_mask = torch.linspace(0, d_norm.shape[1]-1, d_norm.shape[1]).view(1, d_norm.shape[1]).repeat(batch_size * d_norm.shape[1], 1) <= \
                torch.arange(d_norm.shape[1]).repeat(batch_size, 1).view(batch_size*d_norm.shape[1], 1)
        latency  = (d_norm.repeat(1, d_norm.shape[1]).view(batch_size*d_norm.shape[1], d_norm.shape[1]) * latency_mask.to(pi.device)).sum(1).view(batch_size, d_norm.shape[1]) 
        return latency.sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MLPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMLP.initialize(*args, **kwargs)

#    @staticmethod
#    def beam_search(input, beam_size, expand_size=None,
#                    compress_mask=False, model=None, max_calc_batch_size=4096):
#        assert model is not None, "Provide model"
#        assert not compress_mask, "SDVRP does not support compression of the mask"
#
#        fixed = model.precompute_fixed(input)
#
#        def propose_expansions(beam):
#            return model.propose_expansions(
#                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
#            )
#
#        state = SDVRP.make_state(input)
#
#        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float)
    }


class MLPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(MLPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            self.data = [
                {
                    'loc': torch.FloatTensor(size-1, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
