import os
import csv

import numpy as np
import pandas as pd

PATH_DIR_HERE = os.path.dirname(__file__)
PATH_DIR_DATA = "../../data"
PATH_DATA_QUALBANK = os.path.join(PATH_DIR_HERE, PATH_DIR_DATA, "Qualitative_Bankruptcy",
                                  "Qualitative_Bankruptcy.data.txt")


def load_dataset(string):
    string = string.lower()

    if string == "qualbank":
        return load_qualitative_bankruptcy()
    else:
        return None


def load_qualitative_bankruptcy(path_dir_data=PATH_DATA_QUALBANK):
    # Load data into numpy array
    datax = []
    datay = []
    with open(path_dir_data) as file:
        for row in csv.reader(file):
            datax.append(np.array(row[:-1]))
            datay.append(np.array([row[-1]]))
    datax, datay = np.array(datax), np.array(datay)

    # Convert string to an integer enum representation
    # For datax
    datax = (pd.DataFrame(datax).astype('category'))
    categorial_columns = datax.select_dtypes(['category']).columns
    datax[categorial_columns] = \
        datax[categorial_columns].apply(lambda x: x.cat.codes)
    datax = datax.values

    # For datay
    datay = (pd.DataFrame(datay).astype('category'))
    categorial_columns = datay.select_dtypes(['category']).columns
    datay[categorial_columns] = \
        datay[categorial_columns].apply(lambda x: x.cat.codes)
    datay = datay.values

    return datax, datay


if __name__ == '__main__':
    datax, datay = load_dataset("qualbank")
    pd.DataFrame(datax).to_csv("qualbank.csv", header=None, index=None)