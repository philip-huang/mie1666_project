from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.mlp.state_mlp import StateMLP
from problems.mlp.state_mlp_directed import StateMLP_Directed
from utils.beam_search import beam_search
import numpy as np
import ipdb

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


class MLP_S1(object):

    NAME = 'mlp_s1'  # Split Delivery Vehicle Routing Problem


    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['loc'].size()[0], dataset['loc'].size()[-2]
        
        service_times = dataset['service_time']

        a_prev = None

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        pi_with_depot = torch.cat((torch.zeros(batch_size, 1, dtype=torch.uint8, device=pi.device), pi), 1)
        d = loc_with_depot.gather(1, pi_with_depot[..., None].expand(*pi_with_depot.size(), loc_with_depot.size(-1)))

        d_norm = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2) + service_times.gather(1, pi_with_depot)[:, :-1]
        latency_mask = torch.linspace(0, d_norm.shape[1]-1, d_norm.shape[1]).view(1, d_norm.shape[1]).repeat(batch_size * d_norm.shape[1], 1) <= \
                torch.arange(d_norm.shape[1]).repeat(batch_size, 1).view(batch_size*d_norm.shape[1], 1)
        latency  = (d_norm.repeat(1, d_norm.shape[1]).view(batch_size*d_norm.shape[1], d_norm.shape[1]) * latency_mask.to(pi.device)).sum(1).view(batch_size, d_norm.shape[1]) 
        return latency.sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MLPDataset_S1(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMLP_Directed.initialize(*args, **kwargs)


class MLP_S2(object):

    NAME = 'mlp_s2'  # Split Delivery Vehicle Routing Problem


    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['loc'].size()[0], dataset['loc'].size()[-2]
        
        service_times = dataset['service_time']

        a_prev = None

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        pi_with_depot = torch.cat((torch.zeros(batch_size, 1, dtype=torch.uint8, device=pi.device), pi), 1)
        d = loc_with_depot.gather(1, pi_with_depot[..., None].expand(*pi_with_depot.size(), loc_with_depot.size(-1)))

        d_norm = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2) + service_times.gather(1, pi_with_depot)[:, :-1]
        latency_mask = torch.linspace(0, d_norm.shape[1]-1, d_norm.shape[1]).view(1, d_norm.shape[1]).repeat(batch_size * d_norm.shape[1], 1) <= \
                torch.arange(d_norm.shape[1]).repeat(batch_size, 1).view(batch_size*d_norm.shape[1], 1)
        latency  = (d_norm.repeat(1, d_norm.shape[1]).view(batch_size*d_norm.shape[1], d_norm.shape[1]) * latency_mask.to(pi.device)).sum(1).view(batch_size, d_norm.shape[1]) 
        return latency.sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MLPDataset_S2(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMLP_Directed.initialize(*args, **kwargs)


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


def make_instance_directed(args):
    depot, loc, service_time = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float),
        'service_time': torch.tensor(service_time, dtype=torch.float)
    }


class MLPDataset_S1(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(MLPDataset_S1, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance_directed(args) for args in data[offset:offset+num_samples]]

        else:

            nodes = list(range(size))
            self.data = []
            for i in range(num_samples):
                loc = torch.FloatTensor(size-1, 2).uniform_(0, 1)
                depot = torch.FloatTensor(2).uniform_(0, 1) 
    
                loc_with_depot = torch.cat((depot[None, :], loc),  0)
                distances = {(i, j): torch.hypot(loc_with_depot[i, 0] - loc_with_depot[j, 0], loc_with_depot[i, 1] - loc_with_depot[j, 1]) for i in nodes for j in nodes if i != j}
                #d_norm = (loc_with_depot[1:] - loc_with_depot[:-1]).norm(p=2, dim=1)
                max_service_time = (max(distances.values()) - min(distances.values()))/2
                service_times = torch.rand(size-1) * max_service_time
                self.data.append({'loc': loc, 'depot': depot, 'service_time': torch.cat([torch.zeros(1), service_times], dim=0)})

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class MLPDataset_S2(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(MLPDataset_S2, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance_directed(args) for args in data[offset:offset+num_samples]]

        else:

            nodes = list(range(size))
            self.data = []
            for i in range(num_samples):
                loc = torch.FloatTensor(size-1, 2).uniform_(0, 1)
                depot = torch.FloatTensor(2).uniform_(0, 1) 
    
                loc_with_depot = torch.cat((depot[None, :], loc),  0)
                distances = {(i, j): torch.hypot(loc_with_depot[i, 0] - loc_with_depot[j, 0], loc_with_depot[i, 1] - loc_with_depot[j, 1]) for i in nodes for j in nodes if i != j}
                #d_norm = (loc_with_depot[1:] - loc_with_depot[:-1]).norm(p=2, dim=1)
    
                tmax, tmin = max(distances.values()), min(distances.values())
                min_service_time = (tmax + tmin)/2
                max_service_time = (3*tmax - tmin)/2
                service_times = min_service_time + torch.rand(size-1) * (max_service_time - min_service_time)
                self.data.append({'loc': loc, 'depot': depot, 'service_time': torch.cat([torch.zeros(1), service_times], dim=0)})


#            self.data = [
#                {
#                    'loc': loc,
#                    'depot': depot,
#                    'service_time': torch.cat([torch.zeros(1), service_times], dim=0)
#                }
#                for i in range(num_samples)
#            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

