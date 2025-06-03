from django.conf import settings
import os
import pandas as pd
def start_eda_process():
    path = os.path.join(settings.MEDIA_ROOT, 'PaySimDataset.csv')
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns
    import missingno as msno

    import plotly.express as px

    plt.style.use('ggplot')

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    from sklearn.ensemble import RandomForestClassifier
    from lightgbm.sklearn import LGBMClassifier
    import xgboost as xgb
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_curve, RocCurveDisplay

    import warnings
    warnings.filterwarnings('ignore')
    fraud = pd.read_csv(path)
    fraud.head()
    fraud.info()
    fraud.shape
    plt.figure(figsize=(15, 8))
    msno.bar(fraud, figsize=(15, 5), sort='ascending', color="#896F82")
    plt.show()
    print('Number of duplicates are : ', fraud.duplicated().sum())
    fraud.columns
    fraud = fraud.rename(
        columns={'nameOrig': 'origin', 'oldbalanceOrg': 'sender_old_balance', 'newbalanceOrig': 'sender_new_balance',
                 'nameDest': 'destination', 'oldbalanceDest': 'receiver_old_balance',
                 'newbalanceDest': 'receiver_new_balance', 'isFraud': 'isfraud'})
    fraud = fraud.drop(columns=['step', 'isFlaggedFraud'], axis='columns')
    cols = fraud.columns.tolist()
    new_position = 3

    cols.insert(new_position, cols.pop(cols.index('destination')))
    fraud = fraud[cols]
    fraud.head()
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(data=fraud, x="type", hue="isfraud", palette='Set1')
    plt.title('Fraud and Non Fraud Transactions')
    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.01, p.get_height() + 10000))
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
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.countplot(x=fr)
    plt.title('Fraud', fontweight="bold", size=20)
    plt.subplot(1, 2, 2)
    sns.countplot(x=va)
    plt.title('Valid', fontweight="bold", size=20)
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(data=data, x="type")
    plt.title('Transactions according to type')
    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.01, p.get_height() + 10000))

    plt.figure(figsize=(15, 8))
    colors = ['#006400', '#008000', '#00FF00', '#2E8B57', '#2F4F4F']
    plt.pie(data.type.value_counts().values, labels=data.type.value_counts().index, colors=colors, autopct='%.0f%%')
    plt.title("Transactions according to type")
    plt.show()
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(data=data, x="type2", hue="isfraud", palette='Set1')
    plt.title('Transactions according to type2')
    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.01, p.get_height() + 10000))
    plt.figure(figsize=(15, 8))
    colors = ['#006400', '#008000']
    plt.pie(data.type2.value_counts().values, labels=data.type2.value_counts().index, colors=colors, autopct='%.0f%%')
    plt.title("Transactions according to type2")
    plt.show()

    data = pd.get_dummies(data, prefix=['type', 'type2'], drop_first=True)
    X = data.drop('isfraud', 1)
    y = data.isfraud

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=data.isfraud)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rfc = RandomForestClassifier(n_estimators=15, n_jobs=-1, random_state=42)
    lgbm = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=8888)
    xgbr = xgb.XGBClassifier(max_depth=3, n_jobs=-1, random_state=42, learning_rate=0.1)
    logreg = LogisticRegression(solver='liblinear', random_state=42)

    rfc.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    xgbr.fit(X_train, y_train)
    logreg.fit(X_train, y_train)

    classifiers = []
    classifiers.append(rfc)
    classifiers.append(lgbm)
    classifiers.append(xgbr)
    classifiers.append(logreg)
    accuracy_list = []
    auc_list = []

    for classifier in classifiers:
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        accuracy_list.append(accuracy_score(y_test, y_pred))
        auc_list.append(roc_auc_score(y_test, y_pred_proba))

    accuracy_dict = {}
    auc_dict = {}
    for i in range(4):
        key = ['Random Forest', 'Light GBM', 'XGBoost', 'LR'][i]
        accuracy_dict[key] = accuracy_list[i]
        auc_dict[key] = auc_list[i]

    accuracy_dict_sorted = dict(sorted(accuracy_dict.items(), key=lambda item: item[1]))
    auc_dict_sorted = dict(sorted(auc_dict.items(), key=lambda item: item[1]))

    def px_bar(x, y, text, title, color, color_discrete_sequence):
        return px.bar(x=x, y=y, text=text, title=title, color=color, color_discrete_sequence=color_discrete_sequence)

    fig = px_bar(list(accuracy_dict_sorted.keys()), list(accuracy_dict_sorted.values()),
                 np.round(list(accuracy_dict_sorted.values()), 3), 'Accuracy score of each classifiers',
                 list(accuracy_dict_sorted.keys()), px.colors.sequential.matter)
    for idx in [2, 3]:
        fig.data[idx].marker.line.width = 3
        fig.data[idx].marker.line.color = "black"
    fig.show()
    fig = px_bar(list(auc_dict_sorted.keys()), list(auc_dict_sorted.values()),
                 np.round(list(auc_dict_sorted.values()), 3), 'AUC score of each classifiers',
                 list(auc_dict_sorted.keys()), px.colors.sequential.matter)

    for idx in [2, 3]:
        fig.data[idx].marker.line.width = 3
        fig.data[idx].marker.line.color = "black"
    fig.show()
    rfc = RandomForestClassifier(n_estimators=15, n_jobs=-1, random_state=42)
    rfc.fit(X_train, y_train)

    rfc_pred = rfc.predict(X_test)
    rfc_pred_proba = rfc.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, rfc_pred, target_names=['Not Fraud', 'Fraud']))
    fpr, tpr, temp = roc_curve(y_test, rfc_pred_proba)
    auc = round(roc_auc_score(y_test, rfc_pred_proba), 3)
    plt.figure(figsize=(15, 7))
    plt.plot(fpr, tpr, label='Random Forest Classifier, AUC=' + str(auc), linestyle='solid', color='#800000')
    plt.plot([0, 1], [0, 1], color='g')
    plt.title('ROC Curve')
    plt.legend(loc='upper right')
