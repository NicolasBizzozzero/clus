import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score

from clus.src.utils.decorator import remove_unexpected_arguments

ALIASES_ADJUSTED_RAND_INDEX = ("ari", "adjusted_rand_index")


class UnknownEvaluationMetric(Exception):
    def __init__(self, metric_name):
        Exception.__init__(self, "The evaluation metric \"{metric_name}\" doesn't exists".format(
            metric_name=metric_name
        ))


def evaluate(metric, file_affectations_true, file_affectations_pred, name_affectations_true, name_affectations_pred):
    metric_fct = _str_to_evaluation_metric(metric)
    return metric_fct(file_affectations_true=file_affectations_true, file_affectations_pred=file_affectations_pred,
                      name_affectations_true=name_affectations_true, name_affectations_pred=name_affectations_pred)


@remove_unexpected_arguments
def adjusted_rand_index(file_affectations_true, file_affectations_pred, name_affectations_true, name_affectations_pred):
    affectations_true = np.load(file_affectations_true)[name_affectations_true]
    affectations_pred = np.load(file_affectations_pred)[name_affectations_pred]
    return adjusted_rand_score(affectations_true, affectations_pred)


def _str_to_evaluation_metric(string):
    global ALIASES_ADJUSTED_RAND_INDEX

    string = string.lower()
    if string in ALIASES_ADJUSTED_RAND_INDEX:
        return adjusted_rand_index
    raise UnknownEvaluationMetric(string)


if __name__ == "__main__":
    pass
