import enum


@enum.unique
class FusionMethod(enum.IntEnum):
    UNKNOWN = -1
    ITERATIVE = 0
    END_HIERARCHIC_CLUSTERING = 1


class UnknownFusionMethod(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The fusion method : \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def clusters_fusion():
    pass


def int_to_fusionmethod(integer):
    if integer in [method.value for method in FusionMethod]:
        return FusionMethod(integer)
    else:
        raise UnknownFusionMethod(str(integer))


def str_to_fusionmethod(string):
    string = string.lower()
    try:
        string = int(string)
    except ValueError:
        raise UnknownFusionMethod(string)

    return int_to_fusionmethod(string)


def fusionmethod_to_str(method):
    return method.name.lower()


def fusionmethod_to_function(method):
    pass


def int_to_fusionmethod_function(integer):
    method = int_to_fusionmethod(integer)
    return fusionmethod_to_function(method)


def str_to_fusionmethod_function(string):
    method = str_to_fusionmethod(string)
    return fusionmethod_to_function(method)


if __name__ == "__main__":
    pass
