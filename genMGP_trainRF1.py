# source activate drug-class -> rdkit environment
# Though I'm not sure rdkit is actually being used until the end
# of that notebook

import pandas as pd
import numpy as np

import os

#from rdkit import Chem
#from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing

# 1.3 Regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

#from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

tdf = pd.read_csv("VCAP_6h5uM_8496x979_plusFP_4RF.csv", index_col=0)

# Make X_data and y_data from the matrix
X_data = tdf["FP"].tolist()
X_data = map(lambda x: list(x), X_data)
X_data = np.array(list(X_data))
X_data = X_data.astype(float)

features = X_data

# Get the 'labels'
labelendindex = len(tdf.columns.values)-1
label_name_list = tdf.columns.values[0:labelendindex]
labels = tdf[label_name_list].values.tolist()
labels = np.array(labels)

# scale data
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler().fit(labels)
print(transformer)
rl = transformer.transform(labels)
print(rl)

#Split train/test
from sklearn.model_selection import train_test_split
from sklearn.dummy import *

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.10, random_state = 1)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Create models
ESTIMATORS = {
    "Dummy": DummyRegressor(strategy='mean', constant=None, quantile=None),
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,     # Out of 20000
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),                          # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(random_state=0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=1000,
                                                   max_depth=6,
                                                   min_samples_leaf=4,
                                                   random_state=2),
    "Decision Tree Regressor":DecisionTreeRegressor(max_depth=6),
    "MultiO/P GBR" :MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5)),
    "MultiO/P AdaB" :MultiOutputRegressor(AdaBoostRegressor(n_estimators=5))
}

# 9.1 Create an empty dictionary to collect prediction values
y_test_predict = dict()
y_mse = dict()

### make dummy regressor
dr = DummyRegressor(strategy='median', constant=None, quantile=None)
### random forest regressor
rf = RandomForestRegressor(n_estimators = 1000,
                           random_state = 69, n_jobs=1, min_samples_leaf = 10,
                           verbose=3, max_features="log2")
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={
        'max_depth': range(3, 7),
        'n_estimators': (100, 1000, 4000, 8000),
        'max_features': ("auto","log2")},
    cv=5, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)
grid_result = gsc.fit(features, rl)

rfr = grid_result.best_estimator_
rfr.n_jobs=-1
rfr.fit(X_train, y_train)
rf_pred = rfr.predict(X_test)
rf_error = abs(rf_pred - y_test)
print('Mean Absolute Error:', round(np.mean(rf_error), 4), 'z')

dr.fit(X_train,y_train)
dummypred = dr.predict(X_test)
dummyerrors = abs(dummypred - y_test)
print('Mean dummy Absolute Error:', round(np.mean(dummyerrors), 4), 'z')

#Write out data
pd.DataFrame(y_test).to_csv("y_test.csv")
pd.DataFrame(rf_pred).to_csv("rf_pred.csv")
pd.DataFrame(dummypred).to_csv("dummypred.csv")

#write out best RF model params
with open("rfr_test_out.txt", "w") as outfile:
    outfile.write(str(rfr)) 

