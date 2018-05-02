import numpy as np


def SMAPE(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))
