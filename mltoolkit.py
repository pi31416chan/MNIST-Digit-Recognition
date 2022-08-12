# Imports
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import _OneToOneFeatureMixin,TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler,MinMaxScaler



# Classes
class MyScaler(_OneToOneFeatureMixin,TransformerMixin,BaseEstimator):
    def __init__(self,scalertype:str=None):
        '''
        scalertype: str | {'std','minmax','skip'}, Default: None
        specifying the scaler to transform the data
        '''
        if scalertype: 
            if scalertype in ['std','minmax','skip']:
                self.scalertype = scalertype
            else:
                raise ValueError('Specify scaler correctly. Options:{"std","minmax","skip"}')
        else: self.scalertype = None
    def fit(self,X,y=None,scalertype:str=None):
        '''
        scalertype: str | {'std','minmax','skip'}, Default: None
        specifying the scaler to transform the data
        '''
        if scalertype:
            if self.scalertype != scalertype:
                self.scalertype = scalertype
        if self.scalertype == 'std':
            self.scaler = StandardScaler()
        elif self.scalertype == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scalertype == 'skip':
            return self
        else:
            raise ValueError('Specify scaler correctly. Options:{"std","minmax","skip"}')
        return self
    def transform(self,X,scalertype:str=None):
        '''
        scalertype: str | {'std','minmax'}, Default: None
        specifying the scaler to transform the data
        '''
        if scalertype:
            if self.scalertype != scalertype:
                self.scalertype = scalertype
        if self.scalertype == 'std':
            self.scaler = StandardScaler()
        elif self.scalertype == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scalertype == 'skip':
            if type(X) == pd.DataFrame:
                return X.values
            elif type(X) == np.ndarray:
                return X
        else:
            raise ValueError('Specify scaler correctly. Options:{"std","minmax","skip"}')
        return self.scaler.fit_transform(X)

class ImageAugmentor(BaseEstimator,TransformerMixin):
    def __init__(self,rows,columns):
        self.r = rows
        self.c = columns
        self.l = rows * columns
        self.r_zero = np.zeros((1,self.c))
        self.c_zero = np.zeros((self.r,1))
        self._augmented_images = []
        self._augmented_labels = []
        self._X = None
        self._X_fitted = False
        self._y = None
        self._y_fitted = False
        return None
    def fit(self,X,y=None):
        if type(X) == pd.DataFrame:
            self._X = X.values
            self._X_fitted = True
        elif type(X) == np.ndarray:
            self._X = X
            self._X_fitted = True
        else:
            raise ValueError('Input X must be in DataFrame or 2D ndarray.')
        
        if type(y) == pd.Series:
            self._y = y.values
            self._y_fitted = True
        elif type(y) == np.ndarray:
            self._y = y
            self._y_fitted = True
        elif not y:
            pass
        else:
            raise ValueError('Input y must be in Series or 1D ndarray.')
        
        return self
    def transform(self,X,y=None):
        if self._X_fitted == False:
            raise ValueError('Please fit the data before transforming.')
        self._augmented_images = []
        self._augmented_labels = []
        
        for i in range(self._X.shape[0]):
            x = self._X[i].reshape((self.r,self.c))
            if self._y_fitted: label = self._y[i]
            
            self._augmented_images.append(np.concatenate((x[:,1:],self.c_zero),axis=1).flatten())
            self._augmented_images.append(np.concatenate((self.c_zero,x[:,:-1]),axis=1).flatten())
            self._augmented_images.append(np.concatenate((x[1:,:],self.r_zero),axis=0).flatten())
            self._augmented_images.append(np.concatenate((self.r_zero,x[:-1,:]),axis=0).flatten())
            if self._y_fitted: [self._augmented_labels.append(label) for i in range(4)]
        if not self._y_fitted:
            return np.concatenate((self._X,np.asarray(self._augmented_images)),axis=0)
        elif self._y_fitted:
            return np.concatenate((self._X,np.asarray(self._augmented_images)),axis=0),np.concatenate((self._y,np.asarray(self._augmented_labels)))



# Functions
def dump_model(estimator,path = "Trained Models\\",filename=None,yhat=None,scores=None,compress=5):
    '''
    Dump the objects passed as arguments into .pkl file.
    '''
    os.mkdir(path)
    # Dump estimator
    if not filename: filename = str(estimator)
    joblib.dump(estimator,path+filename+".pkl",compress=compress)
    # Dump yhat
    if len(yhat):
        joblib.dump(yhat,path+filename+"_yhat"+".pkl",compress=compress)
    # Dump cv scores
    if len(scores):
        joblib.dump(scores,path+filename+"_scores"+".pkl",compress=compress)



# Test MyScaler
# X = np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9]]).reshape((3,3))
# print(X)
# print()
# sc = MyScaler('std')
# print(sc.transform(X))
# print()
# print(sc.transform(X,'minmax'))
# print()
# print(sc.transform(X,'skip'))
# print()
# print(sc.transform(X))
# print()

# Test ImageAugmentor
# X = np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9]]).reshape((1,9))
# y = np.asarray([1])
# im = ImageAugmentor(3,3)
# X_new,y_new = im.fit_transform(X,y)
# print(np.round(X_new,1))
# print()
# print(y_new)

# Test dump_model
# E = 'Test Estimator'
# y = 'Test yhat'
# s = 'Test scores'
# dump_model(E,yhat=y,scores=s)
# print(joblib.load("Trained Models\\Test Estimator.pkl"))
# print(joblib.load("Trained Models\\Test Estimator_scores.pkl"))
# print(joblib.load("Trained Models\\Test Estimator_yhat.pkl"))