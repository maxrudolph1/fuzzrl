#!/usr/bin/env python3

from libfuzzer import CreateLibFuzzerCounters, LLVMFuzzerRunDriver, LLVMFuzzerMutate
import os
import numpy as np
import sys
import simplejpeg
Counters = CreateLibFuzzerCounters(4096)
def test_custom(a):
    if np.random.random() < 0.5:
        for i in range(100000):
            pass
    else:
        return 3
    return 0

def TestOneInput(a):
    # Instrument the code manually.
    l = len(a)
    if l == 0:
        Counters[0] += 1
    elif l == 8:
        Counters[1] += 1
    elif l == 16:
        Counters[2] += 1
        os.abort()
    else:
        Counters[3] += 1
    
    Counters[4] += 1
    encoded = simplejpeg.encode_jpeg(a, 85, colorsubsampling='422')
    return 0

def TestMe(input):
    # Instrument the code automatically.
    
    return 0

def Initialize(argv):
    return 0

def Mutator(data, max_size, seed):
    a = LLVMFuzzerMutate(data, max_size)
    print(a)
    return a

def CrossOver(data1, data2, out, seed):
    return 0

# If you are using -fork=1, make sure run it like `python3 ./example.py` or
# `./example.py` instead of `python3 example.py`.
LLVMFuzzerRunDriver(sys.argv, TestOneInput, Initialize, Mutator, CrossOver, Counters)
