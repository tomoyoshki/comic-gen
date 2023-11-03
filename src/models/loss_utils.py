import torch

def mask_correlated_samples(batch_size):
        """
        Mark diagonal elements and elements of upper-right and lower-left diagonals as 0.
        """
        N = 2 * batch_size
        diag_mat = torch.eye(batch_size)
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
        mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat
        mask = mask.bool()
        return mask
