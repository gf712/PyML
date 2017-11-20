from ..utils.parsers import csv_parser
from ..maths import transpose


def load_iris():
    """
    Load data from iris dataset
    :return:
    """
    # parse .csv file
    iris_dict = csv_parser('datasets/iris_dataset.csv', column_names=['sepal length in cm',
                                                                      'sepal width in cm',
                                                                      'petal length in cm',
                                                                      'petal width in cm',
                                                                      'class'])

    X = transpose([iris_dict[x] for x in ['sepal length in cm', 'sepal width in cm',
                                          'petal length in cm', 'petal width in cm']])

    y = iris_dict['class']

    return X, y
