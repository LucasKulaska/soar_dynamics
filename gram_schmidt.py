import numpy as np

class GramSchmidt:
    def __init__(self):
        pass
    
    @staticmethod
    def coefficients(Y, v, normalized = True):
        # Each component is the inner product of v by a collumn of Y. Results a collumn vector
        dot_product = Y.T @ v

        # If Y isn't orthonormal, then it makes the inverse of the inner product of each collumn vector by itself.
        if not normalized:
            inv_norm = np.zeros(Y.shape[1])
            for i, x in enumerate(Y.T):
                inv_norm[i] = 1 / np.dot(x,x)
            return np.multiply(dot_product, inv_norm)
        # Else, it returns the dot_product
        return dot_product

    @staticmethod
    def orthonomal_basis(X):
        Y = np.zeros_like(X)
        for i, x in enumerate(X.T):
            coef = GramSchmidt.coefficients(Y, x, normalized = True)
            x -= Y @ coef
            Y[:,i] = x / np.linalg.norm(x)
        return Y