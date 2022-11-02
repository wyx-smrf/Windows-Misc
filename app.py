import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split

st.title("Learning XGBoost")
st.markdown('---')

#%%  Load data from a dataset
def read_data(title, file):
    st.markdown(f'### {title}')
    data = pd.read_csv(file)
    return data

churn_df = read_data('Churn Dataset', 'churn_data.csv')
st.dataframe(churn_df)

#%% Assign the target and its features
select_target = st.selectbox('Pick the target', churn_df.columns)

def target_and_features(dataframe, selectbox):
    # Separate the features and the target
    target = dataframe[f'{selectbox}']
    features = dataframe.drop([f'{selectbox}'], axis=1)

    # Display the features and the target
    col1, col2 = st.columns([2.5,1])
    
    with col1:
        st.markdown('### Features')
        st.dataframe(features)
    with col2:
        st.markdown('### Target')
        st.dataframe(target)

    return features, target

X, y = target_and_features(churn_df, select_target)

#%% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

st.write('Splitting the data into training and testing sets happens automatically.')
# ------------


select_model = st.selectbox('Choose the ML algorithm', ['XGBoost'])

def add_parameters(model):
    params = dict()
    if model == 'XGBoost':
        objective = st.sidebar.selectbox('Select objective', ['binary:logistic']) #, 'multi:softprob'])
        params['objective'] = objective

        n_estimators = st.sidebar.slider('Pick a number', 0, 50, 10)
        params['n_estimators'] = n_estimators

        seed = st.sidebar.number_input('Pick a number', 0, 10)
        params['seed'] = seed

    return params

    
model_params = add_parameters(select_model)



def get_classifier(model, parameters, X_train, X_test, y_train, y_test):
    if model == 'XGBoost':
        clf = xgb.XGBClassifier(
            objective = parameters['objective'],
            n_estimators = parameters['n_estimators'],
            seed = parameters['seed'])
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

    return clf, predictions

model, predictions = get_classifier(select_model, model_params, X_train, X_test, y_train, y_test)

accuracy = float(np.sum(predictions==y_test))/y_test.shape[0]

lastcol1, lastcol2 = st.columns([0.5,2])

with lastcol1:
    st.dataframe(predictions)

with lastcol2:
    st.write("## Accuracy: %f" % (accuracy))




# def eval_classifier(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)

#     return predictions

# results = eval_classifier(model, X_train, X_test, y_train, y_test)


    



