from django.conf import settings
import os
import pandas as pd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns
# import missingno as msno

# import plotly.express as px
from sklearn.metrics import classification_report

plt.style.use('ggplot')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, RocCurveDisplay
import warnings

warnings.filterwarnings('ignore')
path = os.path.join(settings.MEDIA_ROOT, 'PaySimDataset.csv')
fraud = pd.read_csv(path, nrows=1000)
fraud = fraud.rename(
    columns={'nameOrig': 'origin', 'oldbalanceOrg': 'sender_old_balance', 'newbalanceOrig': 'sender_new_balance',
             'nameDest': 'destination', 'oldbalanceDest': 'receiver_old_balance',
             'newbalanceDest': 'receiver_new_balance', 'isFraud': 'isfraud'})
fraud = fraud.drop(columns=['step', 'isFlaggedFraud'], axis='columns')
cols = fraud.columns.tolist()
new_position = 3
cols.insert(new_position, cols.pop(cols.index('destination')))
fraud = fraud[cols]
transfer_fraud = fraud[((fraud['type'] == 'TRANSFER') & fraud['isfraud'] == 1)]
transfer_fraud['origin'].value_counts()
cash_out_fraud = fraud[(fraud['type'] == 'CASH_OUT') & (fraud['isfraud'] == 1)]
cash_out_fraud['destination'].value_counts()
fraud_trans = fraud[fraud['isfraud'] == 1]
valid_trans = fraud[fraud['isfraud'] == 0]

trans_transfer = fraud[fraud['type'] == 'TRANSER']
trans_cashout = fraud[fraud['type'] == 'CASH_OUT']

print('Has the receiving accoung used for cashing out?')
trans_transfer.destination.isin(trans_cashout.origin).any()
data = fraud.copy()
data['type2'] = np.nan
data.loc[fraud.origin.str.contains('C') & fraud.destination.str.contains('C'), 'type2'] = 'CC'
data.loc[fraud.origin.str.contains('C') & fraud.destination.str.contains('M'), 'type2'] = 'CM'
data.loc[fraud.origin.str.contains('M') & fraud.destination.str.contains('C'), 'type2'] = 'MC'
data.loc[fraud.origin.str.contains('M') & fraud.destination.str.contains('C'), 'type2'] = 'MM'
cols = data.columns.tolist()
new_position = 1

cols.insert(new_position, cols.pop(cols.index('type2')))
data = data[cols]
data.drop(columns=['origin', 'destination'], axis='columns', inplace=True)
data.head()
fraud_trans = data[data['isfraud'] == 1]
valid_trans = data[data['isfraud'] == 0]

print('Number of fraud transactions according to type are below:\n', fraud_trans.type2.value_counts(), '\n')
print('Number of valid transactions according to type are below:\n', valid_trans.type2.value_counts())
fr = fraud_trans.type2.value_counts()
va = valid_trans.type2.value_counts()

data = pd.get_dummies(data, prefix=['type', 'type2'], drop_first=True)
X = data.drop('isfraud', 1)
y = data.isfraud

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=data.isfraud,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def start_ml_models():
    print("Random Forest Code")
    rfc = RandomForestClassifier(n_estimators=15, n_jobs=-1, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rf_cr = classification_report(y_test, y_pred, output_dict=True)
    rf_cr = pd.DataFrame(rf_cr).transpose()
    rf_cr = pd.DataFrame(rf_cr)
    rf_cr = rf_cr.to_html
    print("LGBM Classifier")
    lgbm = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=8888)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    lgbm = classification_report(y_test, y_pred, output_dict=True)
    lgbm = pd.DataFrame(lgbm).transpose()
    lgbm = pd.DataFrame(lgbm)
    lgbm = lgbm.to_html
    print("XG Boost Classifiers")
    xgbr = xgb.XGBClassifier(max_depth=3, n_jobs=-1, random_state=42, learning_rate=0.1)
    xgbr.fit(X_train, y_train)
    y_pred = xgbr.predict(X_test)
    xgbr = classification_report(y_test, y_pred, output_dict=True)
    xgbr = pd.DataFrame(xgbr).transpose()
    xgbr = pd.DataFrame(xgbr)
    xgbr = xgbr.to_html
    print("Logistic Regression")
    logreg = LogisticRegression(solver='liblinear', random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    logreg = classification_report(y_test, y_pred, output_dict=True)
    logreg = pd.DataFrame(logreg).transpose()
    logreg = pd.DataFrame(logreg)
    logreg = logreg.to_html

    print("Decesion Tree")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt = classification_report(y_test, y_pred, output_dict=True)
    dt = pd.DataFrame(dt).transpose()
    dt = pd.DataFrame(dt)
    dt = dt.to_html

    print("K Nearest Neighbour")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn = classification_report(y_test, y_pred, output_dict=True)
    knn = pd.DataFrame(knn).transpose()
    knn = pd.DataFrame(knn)
    knn = knn.to_html

    print("SVM Classifiers")
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm = classification_report(y_test, y_pred, output_dict=True)
    svm = pd.DataFrame(svm).transpose()
    svm = pd.DataFrame(svm)
    svm = svm.to_html

    return rf_cr, lgbm, xgbr, logreg, dt, knn, svm


