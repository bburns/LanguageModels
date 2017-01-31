
"""
benchmark.py
Context manager for timing code.

Usage:

    with benchmark("test 1+1"):
        1+1
        
    # or, if you need the time value

    with benchmark("test 1+1") as b:
        1+1
    print(b.time)

Adapted from http://dabeaz.blogspot.fr/2010/02/context-manager-for-timing-benchmarks.html
Posted at http://stackoverflow.com/a/41408510/243392
"""

# timer - see http://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
from timeit import default_timer as timer

class benchmark(object):
    
    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt
        
    def __enter__(self):
        self.start = timer()
        return self
    
    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s: " + self.fmt + " seconds") % (self.msg, t))
        self.time = t

if __name__=='__main__':
    
    # from benchmark import benchmark
    
    with benchmark("test 1+1"):
        1+1
        
    # if you need the time value
    with benchmark("test 1+1") as b:
        1+1
    print(b.time)
    
    




