import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.title("Learning XGBoost (Regression)")
st.markdown('---')

#%%  Load data from a dataset
def read_data(title, file):
    st.markdown(f'### {title}')
    data = pd.read_csv(file)
    return data

housing_df = read_data('Iowa Housing Dataset', 'iowa_housing.csv')
st.dataframe(housing_df)

#%% Assign the target and its features
select_target = st.selectbox('Pick the target', housing_df.columns)

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

X, y = target_and_features(housing_df, select_target)

#%% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

st.write('Splitting the data into training and testing sets happens automatically.')


select_model = st.selectbox('Choose the ML algorithm', ['XGBoost'])

st.sidebar.title('XGBoost Tree Learners')

def model_parameters(model):
    params = dict()
    if model == 'XGBoost':
        objective = st.sidebar.selectbox('Select objective', ['reg:squarederror'])
        params['objective'] = objective

        n_estimators = st.sidebar.slider('Pick a number', 0, 50, 10)
        params['n_estimators'] = n_estimators

        seed = st.sidebar.select_slider('Pick a size', [None, 123])
        params['seed'] = seed

    return params

model_params = model_parameters(select_model)

st.markdown('#### If you choose a wrong target, error will be thrown')



def get_classifier(model, parameters, X_train, X_test, y_train, y_test):
    if model == 'XGBoost':
        clf = xgb.XGBRegressor(params=parameters)   
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    return clf, predictions

model, predictions = get_classifier(select_model, model_params, X_train, X_test, y_train, y_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))


lastcol1, lastcol2 = st.columns([0.5,2])

with lastcol1:
    st.dataframe(predictions)

with lastcol2:
    st.write("## RMSE: %f" % (rmse))


st.sidebar.title('XGBoost Linear Learners')

def lbl_parameters(model):
    params = dict()
    if model == 'XGBoost':
        objective = st.sidebar.selectbox('Select objective', ['reg:squarederror'], key=100)
        params['objective'] = objective

        booster = st.sidebar.selectbox('Select booster', ['booster:gblinear'], key=200)
        params['booster'] = booster


    return params

linear_base_params = lbl_parameters(select_model)


# #%% -------------
# st.markdown('---')
# st.title('XGBoost Cross Validation')

# churn_dmatrix = xgb.DMatrix(data=X, label=y)


# st.sidebar.title('Modify Cross Validation Parameters')

# def cv_parameters(model):
#     params = dict()

#     nfold = st.sidebar.slider('Choose the number of folds', 0, 10, 3)
#     params['nfold'] = nfold

#     num_boost_round = st.sidebar.slider('Pick a number', 0, 10, 5, key=100)
#     params['num_boost_round'] = num_boost_round

#     seed = st.sidebar.number_input('Pick a number', 0, 10, 5, key=110)
#     params['seed'] = seed

#     metrics = st.sidebar.selectbox('Choose metrics to be used', ['error', 'auc'])
#     params['metrics'] = metrics

#     return params

# cv_params = cv_parameters(select_model)

# # Combine two dictionaries into one
# def Merge(dict1, dict2):
#     for i in dict2.keys():
#         dict1[i]=dict2[i]
#     return dict1

# overall_params = Merge(model_params, cv_params)

# st.write(overall_params)


# def xgb_cross_validation(d_matrix_train, parameters):
#     cv = xgb.cv(dtrain=d_matrix_train, params=parameters, as_pandas=True)
#     return cv

# validate = xgb_cross_validation(churn_dmatrix, overall_params)


# cv_col1,cv_col2 = st.columns([2,0.5])

# with cv_col1:
#     st.dataframe(validate)

# with cv_col2:
#     st.write(((1-validate["test-rmse-mean"]).iloc[-1]))

# st.write(validate.columns)
# st.markdown('---')

# params = {"objective":"reg:logistic", "max_depth":3}
# cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
#                   nfold=3, num_boost_round=5, 
#                   metrics="error", as_pandas=True, seed=123)

# # Print cv_results
# st.write(cv_results)

# # Print the accuracy
# st.write(((1-cv_results["test-error-mean"]).iloc[-1]))


# # cv_results = xgb.cv(
# #     dtrain=churn_dmatrix, 
# #     params=params, 
# #     nfold=3, 
# #     num_boost_round=5, 
# #     metrics="error", 
# #     as_pandas=True, 
# #     seed=123)







# # def eval_classifier(model, X_train, X_test, y_train, y_test):
# #     model.fit(X_train, y_train)
# #     predictions = model.predict(X_test)

# #     return predictions

# # results = eval_classifier(model, X_train, X_test, y_train, y_test)


    



