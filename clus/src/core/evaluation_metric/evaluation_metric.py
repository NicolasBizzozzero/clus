import numpy as np

from clus.src.core.evaluation_metric.adjusted_rand_index import adjusted_rand_index

ALIASES_ADJUSTED_RAND_INDEX = ("ari", "adjusted_rand_index")


class UnknownEvaluationMetric(Exception):
    def __init__(self, metric_name):
        Exception.__init__(self, "The evaluation metric \"{metric_name}\" doesn't exists".format(
            metric_name=metric_name
        ))


def evaluate(metric, file_affectations_true, file_affectations_pred, name_affectations_true, name_affectations_pred):
    metric_fct = _str_to_evaluation_metric(metric)

    affectations_ground_truth = np.load(file_affectations_true)[name_affectations_true]
    affectations_prediction = np.load(file_affectations_pred)[name_affectations_pred]

    return metric_fct(affectations_ground_truth=affectations_ground_truth,
                      affectations_prediction=affectations_prediction)


def _str_to_evaluation_metric(string):
    global ALIASES_ADJUSTED_RAND_INDEX

    string = string.lower()
    if string in ALIASES_ADJUSTED_RAND_INDEX:
        return adjusted_rand_index
    raise UnknownEvaluationMetric(string)


if __name__ == "__main__":
    pass
