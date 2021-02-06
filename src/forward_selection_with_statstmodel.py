""" Perform forward stepwise feature selection to narrow down predictors in linear regression
 Use k-fold cross-validation method and Root Mean Squared Error (RMSE) for evaluating models.
"""
import copy
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from statsmodels.formula.api import ols


def forward_select(df: pd.DataFrame, response: str, num_folds=5, test_data_frac=0.25) -> list:
    """ Forward selection for linear regression
    :param df: pandas dataframe with all candidate predictors and the response variable
    :param response: response/independent variable
    :param num_folds: number of folds for dividing data to cross validation and train sets
    :param test_data_frac: fraction of input dataframe kept for final testing
    :return: an statsmoodel regression result with optimal predictors.
    Predictors are selected by forward stepwise selection method and evaluated by Root Mean Squared Error
    """
    remaining_predictors = set(df.columns)
    remaining_predictors.remove(response)
    selected_predictors = []
    test_data = df.sample(frac=test_data_frac)
    train_cv = df.drop(test_data.index)
    folds = kfold(train_cv, num_folds)
    best_predictors = []
    while remaining_predictors:
        rmses_candidates = []
        for candidate in remaining_predictors:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected_predictors + [candidate]))
            rmse = calculate_average_error(folds, formula, response)
            rmses_candidates.append((rmse, candidate))
        rmses_candidates.sort(key=lambda pair: -pair[0])
        cv_rmse, best_candidate = rmses_candidates.pop()
        selected_predictors = selected_predictors + [best_candidate]
        remaining_predictors.remove(best_candidate)
        formula = "{} ~ {} + 1".format(response, ' + '.join(selected_predictors))
        regression = ols(formula, train_cv).fit()
        test_rmse = calculate_rmse(regression, test_data, response)
        best_predictors.append((selected_predictors, cv_rmse, test_rmse))
    return best_predictors


def kfold(df: pd.DataFrame, k: int) -> dict:
    """ Form a dictionary of k folds. Each dictionary item is a dataframe
    :param df: input dataframe
    :param k: number of folds
    :return: dictionary of folds.
    """
    folds = {}
    fold_size = int(len(df)/k)
    df_copy = copy.deepcopy(df)
    for i in range(k - 1):
        folds[i] = df_copy.sample(n=fold_size)
        df_copy.drop(folds[i].index, inplace=True)
    folds[k - 1] = df_copy
    assert len(pd.concat([fold for fold in folds.values()], axis=0)) == len(df)
    return folds

def calculate_average_error(
        folds: dict,
        formula: str,
        response: str,
) -> np.ndarray:
    """ Calculate average RMSE of cross validation folds
    :param folds: dictionary of folds
    :param formula: regression formula
    :param response: response variable
    :return: average RMSE of folds
    """
    num_folds = len(folds)
    rmse_list = []
    for k in range(num_folds):
        regression = ols(formula, folds[k]).fit()
        cv_data = pd.concat([fold_df for (fold, fold_df) in folds.items() if fold != k], axis=0)
        rmse_list.append(calculate_rmse(regression, cv_data, response))

    avg_rmse = np.mean(rmse_list)
    return avg_rmse

def calculate_rmse(regression, data: pd.DataFrame, response: str) -> float:
    """ Calculate RMSE of the given regression on input data
    :param regression: regression results from running statsmodel ols
    :param data: dataframe
    :param response: response variable
    :return:
    """
    predicted = regression.predict(data)
    actual = data[f"{response}"]
    error = actual - predicted
    rss = np.square(error.values)
    rmse = np.sqrt(np.sum(rss)/len(error))
    return rmse

if __name__ == '__main__':
    data = datasets.load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    response = "y_var"
    df[response]=data.target
    logging.debug(f"The dataframe: \n {df.head()}")

    # Run forward stepwise selection
    best_predictors = forward_select(df, response, num_folds=5, test_data_frac=0.2)
    best_predictors.sort(key=lambda formula_errors: formula_errors[1])
    best_formula, cv_error, test_errors = best_predictors[0]
    logging.debug(f"Best model (in terms of test RMSE): {best_formula} \n"
                 f" with test RMSE of {test_errors}"
                 )
    logging.debug(f"All models and their cv errors: \n {best_predictors}")

    # Plot the results
    best_predictors.sort(key=lambda formula_error: len(formula_error[0]))
    plt.figure()
    plt.xlabel("Number of predictors")
    plt.ylabel("RMSE")
    x = list(range(1, len(best_predictors) + 1))
    cv_errors = [formula_error[1] for formula_error in best_predictors]
    plt.xticks(ticks=x)
    plt.plot(x, cv_errors, "-rx", label="CV data")

    test_errors = [formula_error[2] for formula_error in best_predictors]
    plt.gcf()
    plt.plot(x, test_errors, "-bo", label="test data")
    plt.legend()
    plt.show()
