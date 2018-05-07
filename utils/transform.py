'''
Useful transformations of date, indexs, etc.
'''
import numpy as np
import config


def date_to_idx(date):
    '''Index in the original dataset.'''
    return (date - config.START_DATE).days


def flatten_first_2_dimensions(x):
    return x.reshape(-1, *x.shape[2:])


def nan_to_mean(x):
    assert x.ndim == 2, 'X should be 2-dimensional.'
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = col_mean[inds[1]]
    return x


def test():
    print(date_to_idx(config.START_DATE))
    print(date_to_idx(config.TODAY))


if __name__ == '__main__':
    test()
