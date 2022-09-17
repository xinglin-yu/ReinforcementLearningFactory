from contextlib import contextmanager
import time


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('%s: %.3f s' % (name, end - start))


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        name = func.__name__
        print("\n***********", name)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('%s: %.3f s' % (name, end - start))
        return result
    return wrapper
