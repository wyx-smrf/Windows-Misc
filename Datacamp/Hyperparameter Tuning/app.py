import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

credit_card = pd.read_csv('credit-card-full.csv')

st.title('Hyperparameter Tuning')

st.markdown('### Create dummies')



#%% Features and target
# Features and target selection
col1, col2 = st.columns([2,1.5])
with col1:
    feats_to_drop = st.multiselect('Finalize your features (X), delete the target/s or useless column/s:', credit_card.columns)

with col2:
    target = st.selectbox('Select the target', credit_card.columns)

# Features and target confirmation
st.write(f'Are you sure to drop the {feats_to_drop} and create the features (X)?')
st.write(f'As well as to use {target} as the target')
feats_convert_button = st.button('Convert')

# Features and target transformation
def set_target_and_features(dataframe, columns, target):
    df_feats = dataframe.drop(columns=columns, axis=1)
    df_target = dataframe[target]
    return df_feats, df_target

# Features and target presentation
if feats_convert_button:
    col1, col2 = st.columns([2,1])
    X, y = set_target_and_features(credit_card, feats_to_drop, target)
    
    with col1:
        st.write('Features')
        st.dataframe(X)

    with col2: 
        st.write('Target')
        st.dataframe(y)

else:
    pass

st.markdown('---')
#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)




# st.markdown('---')


# col1, col2 = st.columns([2,1])

# with col1:
#     st.write('Here are the new features')
#     st.dataframe(X)

# with col2:

    


# if feats_confirmation: st.write('features are created')








# st.selectbox('Choose', credit_card.columns)

# st.selectbox('Encode categorical variables using:', ['sklearn one-hot encoding', 'pd.get_dummies'])