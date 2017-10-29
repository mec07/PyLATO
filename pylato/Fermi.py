import numpy as np


def fermi_non0(e, mu, kT):
    """Evaluate the Fermi function."""
    x = (e-mu)/kT
    f = np.zeros(x.size, dtype='double')
    for i in range(0, x.size):
        if x[i] > 35.0:
            f[i] = 0.0
        elif x[i] < -35.0:
            f[i] = 1.0
        else:
            f[i] = 1.0/(1.0 + np.exp(x[i]))
    return f


def fermi_0(e, mu, kT):
    """Evaluate the Fermi function."""
    x = e-mu
    f = np.zeros(x.size, dtype='double')
    for i in range(0, x.size):
        if x[i] > 0.0:
            f[i] = 0.0
        elif x[i] < 0.0:
            f[i] = 1.0
        else:
            f[i] = 0.5
    return f
