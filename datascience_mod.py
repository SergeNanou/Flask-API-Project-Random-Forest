# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# Libraries Importing

# %% [code]
#importing libraries

import warnings
import statsmodels.api as sm
import seaborn as sns
warnings.filterwarnings("ignore")
import matplotlib  
import statsmodels.formula.api as smf    
import statsmodels.api as sm 
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import robust_scale
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, linear_model
## for explainer    
from lime import lime_tabular
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import minmax_scaling
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer
from scipy.stats import skew
pd.set_option('display.max_rows', 1000)
## for data
import pandas as pd
import numpy as np
import pickle
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits = 5, random_state = 2)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_log_error

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# Dataset taking with Pandas

# %% [code]
df_train= pd.read_csv('mercedes-benz-greener-manufacturing/train.csv.zip')
df_test= pd.read_csv('mercedes-benz-greener-manufacturing/test.csv.zip')
sub= pd.read_csv('mercedes-benz-greener-manufacturing/sample_submission.csv.zip')

# %% [markdown]
# Data Exploration

# %% [code]
print('Size of training set: {} rows and {} columns'.format(*df_train.shape))
df_train.head()

# %% [code]
df_train.describe()

# %% [markdown]
# Target statistical exploration

# %% [code]
y_train = df_train['y'].values
plt.figure(figsize=(11, 6))
plt.hist(y_train, bins=20)
plt.xlabel('Target value in seconds')
plt.ylabel('Occurences')
plt.title('Distribution of the target value')

print('min: {} max: {} mean: {} std: {}'.format(min(y_train), max(y_train), y_train.mean(), y_train.std()))
print('Count of values above 180: {}'.format(np.sum(y_train > 130)))

# %% [code]
plt.figure(figsize=(15,10))
np.log1p(df_train['y']).hist(bins=100)
plt.xlabel('Log of Y')
plt.title('Distribution of Log of Y variable')
plt.show()

# %% [code]
result = df_train.groupby('y').mean()
plt.figure(figsize=(8,6))
plt.scatter(range(df_train.shape[0]), np.sort(df_train.y.values))

plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()

# %% [markdown]
# Exploration Covariables

# %% [code]
cols = [c for c in df_train.columns if 'X' in c]
print('Feature types:')
df_train[cols].dtypes.value_counts()

# %% [code]

sns.countplot(df_train.dtypes.map(str))
plt.show()

# %% [markdown]
# TOP SELECTION MODEL FEATURE

# %% [code]
for col in df_train.select_dtypes(['object']).columns:
    lb=LabelEncoder()
    lb.fit(df_train[col])
    df_train[col]=lb.transform(df_train[col])
for col in df_test.select_dtypes(['object']).columns:
    lb=LabelEncoder()
    lb.fit(df_test[col])
    df_test[col]=lb.transform(df_test[col])
    
x_train=df_train.drop(['y','ID'],1)
y_train=df_train['y']

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=100)

# %% [code]
fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, height=0.8, ax=ax, max_num_features=30)
plt.show()

# %% [code]
x_train.head()

# %% [markdown]
# Correlation variables selection

# %% [code]
plt.figure(figsize=(13,13))
top_features=['X8','X5','X0','X6','X1','X3','X2','X314','y']
t_1 = ['X8','X5','X0','X6','X1','X3','X2','X314']
train_top_features_df=df_train[top_features]
test_top_features_df=df_test[top_features[0:len(top_features)-1]]
X_train_top_features_df=x_train[['X8','X5','X0','X6','X1','X3','X2','X314']]
sns.heatmap(train_top_features_df.corr(), annot=True)

# %% [markdown]
# Regression linear simple

# %% [code]
# Fit and make the predictions by the model
model = sm.OLS(y_train, X_train_top_features_df).fit()
predictions = model.predict(X_train_top_features_df)

# Print out the statistics
model.summary()

# %% [code]
lm = linear_model.LinearRegression()
model = lm.fit(X_train_top_features_df, y_train)

# %% [code]
lm.score(X_train_top_features_df, y_train)

# %% [code]
# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_train_top_features_df, y_train, test_size=0.3)

# %% [code]
#Import Random Forest Model
from sklearn.ensemble import RandomForestRegressor

#Create a Gaussian Classifier
clf = RandomForestRegressor(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_reg, y_train_reg)

y_pred = clf.predict(X_test_reg)

# %% [code]

# Calculate the absolute errors
errors = abs(y_pred - y_test_reg)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# %% [code]
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test_reg)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %% [code]
# Limit depth of tree to 3 levels
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
rf_small = RandomForestRegressor(n_estimators=9, max_depth = 3)
rf_small.fit(X_train_reg, y_train_reg)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree_1.dot', feature_names = t_1, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree_1.dot')
graph.write_png('small_tree_1.png');

# %% [code]
from IPython.display import Image
Image(filename = 'small_tree.png')

# %% [markdown]
# Pickle Models

# %% [code]
pickle.dump(clf, open('clf.pkl','wb'))

model = pickle.load(open('clf.pkl','rb'))


# %% [code]
