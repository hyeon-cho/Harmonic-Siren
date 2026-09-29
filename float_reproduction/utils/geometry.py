import torch

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def make_coord_3d(shape, ranges=None, flatten=True):
    """Make 3D coordinates at grid centers."""
    assert len(shape) == 3, "Shape must have exactly three dimensions for 3D coordinates."
    coord_seqs = []
    for i in range(3):
        n = shape[i]
        if ranges is None:
            v0, v1 = -1.0, 1.0
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2.0 * n)
        seq = v0 + r + (2.0 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    grids = torch.meshgrid(coord_seqs[0], coord_seqs[1], coord_seqs[2], indexing='ij')
    ret = torch.stack(grids, dim=-1)
    if flatten:
        ret = ret.view(-1, 3)
    return ret
