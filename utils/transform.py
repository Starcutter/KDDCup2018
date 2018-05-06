'''
Useful transformations of date, indexs, etc.
'''
import pandas as pd
import config


def date_to_idx(date):
    '''Index in the original dataset.'''
    return (date - config.START_DATE).days


def flatten_first_2_dimensions(x):
    return x.reshape(-1, *x.shape[2:])


def test():
    print(date_to_idx(config.START_DATE))
    print(date_to_idx(config.TODAY))


if __name__ == '__main__':
    test()
