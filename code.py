# Final-project-code
Final project code(LANL Earthquake Prediction)
#import package
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, make_scorer
warnings.filterwarnings("ignore")

#read in train file
%%time
train = pd.read_csv('../input/train.csv',dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head()

#read in test file
test_path = "../input/test"
test_files = os.listdir("../input/test")
print(test_files[0:5])

#generate features
def gen_features(X):
    fe = []
    fe.append(X.mean())
    fe.append(X.std())
    fe.append(X.min())
    fe.append(X.max())
    fe.append(X.kurtosis())
    fe.append(X.skew())
    fe.append(np.quantile(X,0.01))
    fe.append(np.quantile(X,0.05))
    fe.append(np.quantile(X,0.95))
    fe.append(np.quantile(X,0.99))
    fe.append(np.abs(X).max())
    fe.append(np.abs(X).mean())
    fe.append(np.abs(X).std())
    return pd.Series(fe)
    
#Visualization plotting for train set, test set and submission    
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
#Taking the segment id from sample_submission file
#Applying Feature Engineering on test data files
test = pd.DataFrame()
for seg_id in submission.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    ch = gen_features(seg['acoustic_data'])
    X_test = test.append(ch, ignore_index=True)
    
fig, ax = plt.subplots(4,1, figsize=(20,25))
for n in range(4):
    seg = pd.read_csv(test_path  + test_files[n])
    ax[n].plot(seg.acoustic_data.values, c="mediumseagreen")
    ax[n].set_xlabel("Index")
    ax[n].set_ylabel("Signal")
    ax[n].set_ylim([-300, 300])
    ax[n].set_title("Test {}".format(test_files[n]));
    
fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.distplot(train.signal.values, ax=ax[0], color="Red", bins=100, kde=False)
ax[0].set_xlabel("Signal")
ax[0].set_ylabel("Density")
ax[0].set_title("Signal distribution")

low = train.signal.mean() - 3 * train.signal.std()
high = train.signal.mean() + 3 * train.signal.std() 
sns.distplot(train.loc[(train.signal >= low) & (train.signal <= high), "signal"].values,
             ax=ax[1],
             color="Orange",
             bins=150, kde=False)
ax[1].set_xlabel("Signal")
ax[1].set_ylabel("Density")
ax[1].set_title("Signal distribution without peaks");


train_ad_sample_df = train['acoustic_data'].values[::100]
train_ttf_sample_df = train['time_to_failure'].values[::100]

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)
del train_ad_sample_df
del train_ttf_sample_df

#XGB model
import time
import lightgbm as lgb
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y_train.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, y_train)**0.5))

#NuSVR model
GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
  gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
  tol=0.01, verbose=False),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid=[{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1], 'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}],
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='neg_mean_absolute_error', verbose=0)

#KNN model
knn_final = neighbors.KNeighborsRegressor(500, weights='uniform', metric='manhattan')
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
knn_final.fit(scaler.transform(X_train), y_train)

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(dtype = np.float64, index = submission.index,columns=range(0,1500))
for seg_id in tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    xc = np.sort(seg['acoustic_data'].values)
    X_test.loc[seg_id,:] = np.reshape(xc, (-1, 100)).sum(axis=-1)

submission['time_to_failure'] = knn_final.predict(scaler.transform(X_test))
print(submission.head(5))

submission.to_csv('knn_std.csv')


