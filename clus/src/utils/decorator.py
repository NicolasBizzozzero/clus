import inspect
import functools
from typing import List


def remove_unexpected_arguments(func):
    """ The decorated function silently ignore unexpected parameters without
    raising any error.

    Authors :
    * BIZZOZZERO Nicolas
    * POUYET Adrien
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        possible_parameters: List[str] = inspect.getfullargspec(func).args
        new_kwargs = dict(filter(lambda a: a[0] in possible_parameters, kwargs.items()))

        return func(*args, **new_kwargs)
    return wrapper


if __name__ == "__main__":
    pass
