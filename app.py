import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split

st.title("Learning XGBoost")
st.markdown('---')

churn_data = pd.read_csv('churn_data.csv')
st.markdown('### Churn dataset')
st.dataframe(churn_data)

target = st.selectbox('Pick the target', churn_data.columns)

col1, col2 = st.columns([2,2])

with col1: 
    st.write(f'If you choose {target} as the target, then these are the features')
    y = churn_data[f'{target}']
    X = churn_data.drop([target], axis=1)
    st.write(X.columns)

with col2: 
    if st.button('Split data'):
        
        st.write('Done :smile:')

algo_dict = {'XGBoost': 'xgb.XGBClassifier'}

algo = st.selectbox('Select the ML Algorithm', algo_dict.keys())

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


def xgb_clf(algorithm, X_train, X_test, y_train, y_test):
    if algorithm == 'XGBoost':
        xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
        xg_cl.fit(X_train, y_train)
        preds = xg_cl.predict(X_test)
        accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
        return accuracy
    else:
        pass

st.write(xgb_clf(algo, X_train, X_test, y_train, y_test))

