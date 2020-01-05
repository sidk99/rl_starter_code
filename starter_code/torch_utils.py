def dist_bsize(torch_dist):
    batch_shape = torch_dist.batch_shape
    assert len(batch_shape) == 1 or (len(batch_shape) == 2 and batch_shape[-1] == 1)  # very hacky. Should just write a wrapper for Categorical
    return batch_shape[0]