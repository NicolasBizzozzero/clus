import functools
import inspect
from time import time

from memory_profiler import memory_usage


def remove_unexpected_arguments(func):
    """ The decorated function silently ignore unexpected parameters without
    raising any error.

    Authors :
    * BIZZOZZERO Nicolas
    * POUYET Adrien
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        possible_parameters = inspect.getfullargspec(func).args
        new_kwargs = dict(filter(lambda a: a[0] in possible_parameters,
                                 kwargs.items()))
        return func(*args, **new_kwargs)
    return wrapper


def wrap_max_memory_consumption(func):
    """ Add the maximal memory consumption of a function in MiB to the dictionary it is returning.
    Differences can occurs between executions due to the fact that the information stored by the profiler is counted
    within the total memory used by the program. There is also a slight overhead of storing this information. Also, some
    variations can be seen due to how Python manages memory.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_mem_usage, result = memory_usage(proc=(func, args, kwargs), max_usage=True, retval=True)
        if isinstance(result, dict):
            result["max_memory_usage"] = max_mem_usage[0]
        return result
    return wrapper


def time_this(func):
    """ Print the execution time of the wrapped function. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        time_begin = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_total = time_end - time_begin
        second_or_seconds = "second" if (time_total < 1) else "seconds"
        print("Execution time for \"{}\": {} {}".format(func.__name__,
                                                        time_total,
                                                        second_or_seconds))
        return result
    return wrapper


def error_fallback(error, fallback_function):
    def real_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except error:
                return fallback_function(*args, **kwargs)
        return wrapper
    return real_decorator


if __name__ == "__main__":
    pass
