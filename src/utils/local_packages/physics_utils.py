"""
Local physics constants and utilities - Alternative to missing physics package
"""
import math

# Physical constants
SPEED_OF_LIGHT = 299792458  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J*Hz^-1
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2

def golden_ratio():
    return GOLDEN_RATIO

def fibonacci(n):
    """Generate fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
