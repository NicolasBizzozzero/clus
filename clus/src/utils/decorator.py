import functools
import inspect
from time import time


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
        new_kwargs = dict(filter(lambda a: a[0] in possible_parameters, kwargs.items()))

        return func(*args, **new_kwargs)
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


if __name__ == "__main__":
    pass
