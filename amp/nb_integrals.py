import numba
from numba import cfunc,carray
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from math import exp, erfc, sqrt, pi


def jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx,n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)


@jit_integrand_function
def I0_00(args):
    u, V22, Vb, w1, w2, alpha12 = args
    """I0 for y = (0,0)"""
    expo= exp(-.5*(u**2))/sqrt(2.*pi)
    erf_in= (-u*Vb*w1/alpha12 + V22*w2)/sqrt(2*V22)
    return expo*erfc(erf_in)


@jit_integrand_function
def I1_00(args):
    u, V22, Vb, w1, w2, alpha12 = args
    """I1 for y = (0,0)"""
    expo= exp(-.5*(u**2))/sqrt(2.*pi)
    erf_in= (-u*Vb*w1/alpha12 + V22*w2)/sqrt(2*V22)
    return expo*u*erfc(erf_in)


@jit_integrand_function
def I2_00(args):
    u, V22, Vb, w1, w2, alpha12 = args
    """I1 for y = (0,0)"""
    expo= exp(-.5*(u**2))/sqrt(2.*pi)
    erf_in= (-u*Vb*w1/alpha12 + V22*w2)/sqrt(2*V22)
    return expo*(u**2)*erfc(erf_in)


@jit_integrand_function
def I0_10(args):
    u, V22, Vb, w1, w2, alpha12 = args
    """I1 for y = (1,0) and y = (0,1)"""
    expo= exp(-.5*(u**2))/sqrt(2.*pi)
    erf_in= (-u*(Vb+V22)*w1/alpha12 + V22*(w2-w1))/sqrt(2*V22)
    return expo*erfc(erf_in)


@jit_integrand_function
def I1_10(args):
    u, V22, Vb, w1, w2, alpha12 = args
    """I1 for y = (1,0) and y = (0,1)"""
    expo= exp(-.5*(u**2))/sqrt(2.*pi)
    erf_in= (-u*(Vb+V22)*w1/alpha12 + V22*(w2-w1))/sqrt(2*V22)
    return expo*u*erfc(erf_in)


@jit_integrand_function
def I2_10(args):
    u, V22, Vb, w1, w2, alpha12 = args
    """I1 for y = (1,0) and y = (0,1)"""
    expo= exp(-.5*(u**2))/sqrt(2.*pi)
    erf_in= (-u*(Vb+V22)*w1/alpha12 + V22*(w2-w1))/sqrt(2*V22)
    return expo*(u**2)*erfc(erf_in)
