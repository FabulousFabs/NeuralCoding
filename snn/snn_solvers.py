import numpy as np

def Heuns(x0, t0, dt, dxdt, **kwargs):
    '''
    Heun's method for pedantically solving diffeqs
    '''

    t1 = t0 + dt
    dxdt0 = dxdt(t0, x0, kwargs)
    dxdt1 = dxdt(t1, x0 + dt * dxdt0, kwargs)

    return x0 + (dt / 2) * (dxdt0 + dxdt1)

def Eulers(x0, t0, dt, dxdt, h = None, n = 10, **kwargs):
    '''
    Euler's method for pedantically solving diffeqs
    '''

    if h is None:
        h = dt / n

    for j in np.arange(n):
        m = dxdt(t0, x0, kwargs)
        y1 = x0 + h * m
        t1 = t0 + h

        t0 = t1
        x0 = y1

    return x0

def RungeKutta(x0, t0, dt, dxdt, h = None, n = 10, **kwargs):
    '''
    Runge-Kutta's method for pedantically solving diffeqs
    '''

    if h is None:
        h = dt / n

    for j in np.arange(n):
        k1 = dxdt(t0, x0, kwargs)
        k2 = dxdt(t0 + h/2, x0 + h * (k1 / 2), kwargs)
        k3 = dxdt(t0 + h/2, x0 + h * (k2 / 2), kwargs)
        k4 = dxdt(t0 + h, x0 + h * k3, kwargs)

        x0 = x0 + (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h

    return x0

def Clean(x0, t0, dt, dxdt, **kwargs):
    '''
    Use if you don't want to solve
    '''

    return dxdt(t0, x0, kwargs)
