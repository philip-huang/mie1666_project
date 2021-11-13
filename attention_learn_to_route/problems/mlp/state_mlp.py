import torch
from typing import NamedTuple


class StateMLP(NamedTuple):
    # Fixed input
    coords: torch.Tensor

    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited: torch.Tensor
    latency: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step


    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited=self.visited[key],
            latency=self.latency[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    @staticmethod
    def initialize(input):

        depot = input['depot']
        loc = input['loc']

        batch_size, n_loc, _ = loc.size()
        return StateMLP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited=torch.cat((
                torch.ones(batch_size, 1, 1,
                    dtype=torch.uint8, device=loc.device),
                torch.zeros(batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device)
                ), -1),
            latency=torch.zeros(batch_size, 1, device=loc.device),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.ones(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        latency = self.latency + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        lengths = self.lengths + latency  # (batch_dim, 1)

        visited = self.visited.scatter(-1, prev_a[:, :, None], 1)

        return self._replace(
            prev_a=prev_a, visited=visited, latency=latency,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.visited.size(-1) and not (self.visited == 0).any()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask, depends on already visited
        :return:
        """

        # Every node is visited exactly once
        return self.visited > 0

    def construct_solutions(self, actions):
        return actions
