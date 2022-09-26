import pandas as pd 
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


""" 
====================== Imputación de datos con Complete Case Analysis  ======================
"""
class CompleteCaseAnalysis(BaseEstimator, TransformerMixin):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Operador de ingeniería de caracteristicas para imputación CCA
    """

    def __init__(self, varNames = None):
        self.varNames = varNames
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.dropna(subset=self.varNames, inplace=True)
        return X


""" 
====================== Imputación de datos con Arbitrary Imputation  ======================
"""
class ArbitraryImputation(BaseEstimator, TransformerMixin):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Operador de ingeniería de caracteristicas para imputación arbitraria
    """

    def __init__(self, varNames = None):
        self.varNames = varNames
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.varNames].fillna(0, inplace=True)
        return X


""" 
====================== CategoricalEncoderOperator ======================
"""
class CategoricalEncoderOperator(TransformerMixin, BaseEstimator):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Operador de ingeniería de caracteristicas para codificar las variables categóricas
    """

    def __init__(self, varNames):
        self.varNames =  varNames

    def fit(self, X, y=None):
        self.mapper = {}
        for varname in self.varNames:
            self.mapper[varname] = X[varname].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for varname in self.varNames:
            X[varname] = X[varname].map(self.mapper[varname])        
        return X


""" 
====================== Tratamiento de Outliers ======================
"""

class OutliersTreatmentOperator(BaseEstimator, TransformerMixin):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Operador para el tratamiento de outliers
    """

    def __init__(self, factor = 1.75, varNames=None):
        self.varNames = varNames
        self.factor = factor

    
    def fit(self, X, y=None):
        for col in self.varNames:
            q3 = X[col].quantile(0.75)
            q1 = X[col].quantile(0.25)
            self.IQR = q3 - q1
            self.upper = q3 + self.factor *self.IQR
            self.lower = q1 - self.factor * self.IQR
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.varNames:
            X[col] = np.where(X[col] >= self.upper, self.upper,
                np.where(X[col]< self.lower, self.lower, X[col]))

        return X

