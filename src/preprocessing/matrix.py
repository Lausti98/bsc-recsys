#pylint: skip-file
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix

def ItemInteractionMatrix(X, item_col='item_id', user_col='user_id', rating_col='rating'):
    # Remove duplicate user-item interactions
    X = X[~X[[item_col, user_col]].duplicated()]
    #Get users and items as categories
    user_c = CategoricalDtype(sorted(X[user_col].unique()), ordered=True)
    item_c = CategoricalDtype(sorted(X[item_col].unique()), ordered=True)

    col = X[user_col].astype(user_c).cat.codes
    row = X[item_col].astype(item_c).cat.codes

    csr = csr_matrix((X[rating_col], (row, col)), \
                            shape=(item_c.categories.size, user_c.categories.size))
    return csr