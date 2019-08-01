from sklearn.metrics import adjusted_mutual_info_score, fowlkes_mallows_score, homogeneity_completeness_v_measure, \
    homogeneity_score, mutual_info_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score, completeness_score as sklearn_completeness_score,\
    contingency_matrix as sklearn_contingency_matrix

from clus.src.utils.decorator import remove_unexpected_arguments

ALIASES_ADJUSTED_RAND_INDEX = ("ari", "adjusted_rand_index")
ALIASES_ADJUSTED_MUTUAL_INFO = ("ami", "adjusted_mutual_info")
ALIASES_COMPLETENESS = ("completeness",)
ALIASES_CONTINGENCY_MATRIX = ("contingency_matrix",)
ALIASES_FOWLKES_MALLOWS_INDEX = ("fmi", "fowlkes_mallows_index")
ALIASES_HOMOGENEITY = ("homogeneity",)
ALIASES_MUTUAL_INFO = ("mi", "mutual_info")
ALIASES_NORMALIZED_MUTUAL_INFO = ("nmi", "normalized_mutual_info")
ALIASES_V_MEASURE = ("v", "v_measure")


class UnknownEvaluationMetric(Exception):
    def __init__(self, metric_name):
        Exception.__init__(self, "The evaluation metric \"{metric_name}\" doesn't exists".format(
            metric_name=metric_name
        ))


def evaluate(*, metric, affectations_true, affectations_pred, average_method, eps, sparse, beta):
    metric_fct = _str_to_evaluation_metric(metric)
    return metric_fct(affectations_true=affectations_true, affectations_pred=affectations_pred,
                      average_method=average_method, eps=eps, sparse=sparse, beta=beta)


@remove_unexpected_arguments
def adjusted_rand_index(affectations_true, affectations_pred):
    return adjusted_rand_score(affectations_true, affectations_pred)


@remove_unexpected_arguments
def adjusted_mutual_info(affectations_true, affectations_pred, average_method="arithmetic"):
    return adjusted_mutual_info_score(affectations_true, affectations_pred, average_method=average_method)


@remove_unexpected_arguments
def completeness(affectations_true, affectations_pred):
    return sklearn_completeness_score(affectations_true, affectations_pred)


@remove_unexpected_arguments
def contingency_matrix(affectations_true, affectations_pred, eps=None, sparse=False):
    return sklearn_contingency_matrix(affectations_true, affectations_pred, eps=eps, sparse=sparse)


@remove_unexpected_arguments
def fowlkes_mallows_index(affectations_true, affectations_pred, sparse=False):
    return fowlkes_mallows_score(affectations_true, affectations_pred, sparse=sparse)


@remove_unexpected_arguments
def homogeneity(affectations_true, affectations_pred):
    return homogeneity_score(affectations_true, affectations_pred)


@remove_unexpected_arguments
def mutual_information(affectations_true, affectations_pred):
    return mutual_info_score(affectations_true, affectations_pred, contingency=None)


@remove_unexpected_arguments
def normalized_mutual_information(affectations_true, affectations_pred, average_method="arithmetic"):
    return normalized_mutual_info_score(affectations_true, affectations_pred, average_method=average_method)


@remove_unexpected_arguments
def v_measure(affectations_true, affectations_pred, beta=1.0):
    return v_measure_score(affectations_true, affectations_pred, beta=beta)


def _str_to_evaluation_metric(string):
    global ALIASES_ADJUSTED_RAND_INDEX

    string = string.lower()
    if string in ALIASES_ADJUSTED_RAND_INDEX:
        return adjusted_rand_index
    if string in ALIASES_ADJUSTED_MUTUAL_INFO:
        return adjusted_mutual_info
    if string in ALIASES_COMPLETENESS:
        return completeness
    if string in ALIASES_CONTINGENCY_MATRIX:
        return contingency_matrix
    if string in ALIASES_FOWLKES_MALLOWS_INDEX:
        return fowlkes_mallows_index
    if string in ALIASES_HOMOGENEITY:
        return homogeneity
    if string in ALIASES_MUTUAL_INFO:
        return mutual_information
    if string in ALIASES_NORMALIZED_MUTUAL_INFO:
        return normalized_mutual_information
    if string in ALIASES_V_MEASURE:
        return v_measure
    raise UnknownEvaluationMetric(string)


if __name__ == "__main__":
    pass
