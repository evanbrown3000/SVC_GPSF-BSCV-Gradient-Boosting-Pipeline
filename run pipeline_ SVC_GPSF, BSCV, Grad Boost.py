#Evan Brown Sep 21, 2020
#replaced sklearn's standard SVC with personally upgraded version called SVC_GPSF(Gamma Proportional to Scaling Factor)
#bayesian cross validated search to tune SVC
#subsequent gradient boosting (XGBoost)

from SVC_GPSF import SVC_GPSF #(Gamma Proportional to Scaling Factor)

#*** using sklearn's GradientBoostingRegressor (trees) because tablet is x32 python
#alternatives include LightBoost, CatBoost
#from xgboost import XGBRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import load_digits

X,y = load_digits(return_X_y=1)
X, y = (X[:300],y[:300])	#limit training time for grid search
X_train,X_test,y_train, y_test = train_test_split(X, y, train_size = 0.8)


print('input train shapes: ',X_train.shape, y_train.shape, 'input test shapes:', X_test.shape,y_test.shape)

pipe = Pipeline([
                ('scale' , StandardScaler()), #set scale to passthrough for PCA
                ('dim_reduce', KMeans()),
                ('svc', SVC_GPSF(probability=True))
])

#bayesian search for best cross validated hyperparameters, including trying various dimensionality reduction techniques
 
params_pca = {
'scale' : ['passthrough'],
'dim_reduce': [PCA()], 'dim_reduce__n_components' : Integer(30,40, prior = 'uniform'),
'svc__ratio': Real(.01, 100, prior = 'log-uniform'), 'svc__gamma' : ['scale', 'auto'] , 'svc__C': Real(1, 200, prior = 'log-uniform', base=2)
}
 
params_kmeans = {
'scale' : [StandardScaler()],
'dim_reduce' : [KMeans()], 'dim_reduce__n_clusters' : Integer(30, 40, prior = 'uniform'),
'svc__ratio': Real(.01, 100, prior = 'log-uniform'), 'svc__gamma' : ['scale', 'auto'] , 'svc__C': Real(1, 200, prior = 'log-uniform', base=2)
}
params_pass = {
'scale': ['passthrough'],
'dim_reduce': ['passthrough'],
'svc__ratio': Real(.01, 100, prior = 'log-uniform'), 'svc__gamma' : ['scale', 'auto'] , 'svc__C': Real(1, 200, prior = 'log-uniform', base=2)    
}
bscv = BayesSearchCV( pipe,
# (parameter space, # of evaluations)
[(params_pca, 20), (params_kmeans, 10), (params_pass, 5)], #set # evals to 1 for testing, 40 20 10 for final
cv=3, refit= False
)


X_train.shape, y_train.shape

time_start_bscv =timer()
bscv.fit(X_train, y_train)
time_end_bscv = timer()
print("bayes search cv fit time: " , time_end_bscv- time_start_bscv)
best_params = bscv.best_params_
pipe_best = pipe #rename
pipe_best.set_params(**best_params)

#running through steps of pipeline to manually get X_reduced for additional boosting afterward
#pipe_best.fit(X_train)
#y_pred = pipe_best.predict(X_test)

time_start_pipe = timer()

#fit pipe_best to full train set
X_scaled = None
if(pipe_best.steps[0][1]!='passthrough'):
	X_scaled = pipe_best.steps[0][1].fit_transform(X_train)
else:
	X_scaled = X_train
X_red_train = None
if(pipe_best.steps[1][1]!='passthrough'):
	X_red_train = pipe_best.steps[1][1].fit_transform(X_scaled)
else:
	X_red_train = X_scaled
pipe_best.steps[2][1].fit(X_red_train, y_train)
X_scaled = None

time_end_pipe = timer()
print("pipe fit time: " , time_end_pipe- time_start_pipe)

#predict with pipe best on train set to train boosting with residuals
y_prob = pipe_best.steps[2][1].predict_proba(X_red_train)
y_class = np.argmax(y_prob, axis=1)

#get residuals from train
#for target class residuals only: y_prob[np.linspace(0,60,1,dtype=np.int8) , y_class==y_test]
y_target_prob = np.zeros((len(y_train), len(np.unique(y_train)))) #M obs, classes
y_target_prob[ np.arange(len(y_train)) , y_train ] = 1 #M target probs(binary), classes
residuals = y_target_prob - y_prob

#residuals from training---
#print("mean residual probabilities per class, unboosted:", np.mean(residuals, axis = 0))

#~~~~~~~~~~~~
#boost svc by fitting to residuals an xtreme gradient boosted decision tree regressor and adding output  to predicted class probabilities
#TESTING	NOTE:  using	sklearn's GradientBoostingRegressor (trees) because tablet is x32 python
gbr = GradientBoostingRegressor()
mor_gbr = MultiOutputRegressor(gbr, n_jobs=-1)

time_start_boost = timer()
mor_gbr.fit(X_red_train, residuals) #fit booster to train
time_end_boost = timer()
print("boost fit time: " , time_end_boost- time_start_boost)
#Done fitting booster~~~~


#predict with pipe best~~
if(pipe_best.steps[0][1]!='passthrough'):
	X_scaled = pipe_best.steps[0][1].fit_transform(X_test)
else:
	X_scaled = X_test
if(pipe_best.steps[1][1]!='passthrough'):
	X_red_test = pipe_best.steps[1][1].fit_transform(X_scaled)
else:
	X_red_test = X_scaled
y_prob = pipe_best.steps[2][1].predict_proba(X_red_test)
y_class = np.argmax(y_prob, axis=1)
#~~~~~~

#summing predictions on test~~~~
error_prob = mor_gbr.predict(X_red_test) #predict booster on test
y_prob_boosted = y_prob + error_prob #add model1 test preds and booster test preds
y_class_boosted = np.argmax(y_prob_boosted, axis =1)

#evaluating
print('mse pipeline:', mse(y_test, y_class))
print('accuracy pipeline:', accuracy_score(y_test, y_class))
print('mse pipeline with boosting:', mse(y_test, y_class_boosted))
print('accuracy boosted:', accuracy_score(y_test, y_class_boosted))