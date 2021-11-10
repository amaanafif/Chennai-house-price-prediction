import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

df1 = pd.read_csv("clean_data.csv")

df1["bathroom"].fillna(-1, inplace = True)
df1["age"].fillna(-1, inplace = True)

def bath_finder(x,y):
    if y == -1:
        if x >= 5:
            return x+1
        elif x == 4 | x == 3:
            return  x
        elif x == 1:
            return x
        else:
            return x-1
    else:
        return y
    
df1["bath"] = df1.apply(lambda x: bath_finder(x["bhk"], x["bathroom"]), axis = 1)

def age_finder(x):
    if x == -1:
        return 0
    else:
        return x
df1['year'] = df1['age'].apply(age_finder)

df1.drop(['bathroom','age'],axis=1,inplace=True)

location_stats = df1['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df1.location = df1.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

builder_stats = df1['builder'].value_counts(ascending=False)
builder_stats_less_than_10 = builder_stats[builder_stats<=10]
df1.builder = df1.builder.apply(lambda x: 'other' if x in builder_stats_less_than_10 else x)

df2 = df1[~(df1.area/df1.bhk<300)]
df2['price_per_sqft'] = df2['price']*100000/df2['area']

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3 = remove_pps_outliers(df2)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df4 = remove_bhk_outliers(df3)

df5 = df4.drop(['price_per_sqft'],axis='columns')
df5.head()

dummies1 = pd.get_dummies(df5.location)
dummies2 = pd.get_dummies(df5.builder)
df5.drop(['status'],axis=1,inplace=True)
df5 = pd.concat(((df5,dummies1,dummies2)),axis=1)
df5.drop(['location','builder'],axis=1,inplace=True)

X = df5.drop(['price'],axis=1)
y = df5.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

st.title("Chennai House Price Predictor")
st.subheader("Looking to Buy a House in Chennai")
st.subheader("Predict the Price Here")

a=df4.sort_values('location')
b=a['location'].unique()

with st.form(key="form1"):
    location_input = st.selectbox('Location',b)

    c=df4.sort_values('builder')
    d=c['builder'].unique()
    builder_input = st.selectbox('Builder',d)

    area_input = st.text_input("Enter House Area (Sqft)")

    Bhk = [1, 2, 3, 4, 5, 6, 8]
    b = st.selectbox('Choose the BHK',Bhk)

    bath_input = st.number_input('Choose the Number of Bathroom', min_value=1, max_value=9, value=2, step=1)

    year_input = st.slider('Choose the age of the house', min_value=0, max_value=8, value=0, step=1)


    submit = st.form_submit_button("Submit")

if submit:
    loc_index = np.where(X.columns==location_input)[0][0]
    builder_index = np.where(X.columns==builder_input)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = area_input
    x[1] = b
    x[2] = bath_input
    x[3] = year_input
    if loc_index >= 0:
        x[loc_index] = 1
    if builder_index >= 0:
        x[builder_index] = 1
    final_price = lr_clf.predict([x])[0]

    final_price = round(final_price, 2)
    if final_price>=100:
        final_price=final_price/100
        final_price = round(final_price, 2)
        st.subheader("The predicted price in Crores is")
        st.subheader(final_price)
    else:
        final_price=final_price
        final_price = round(final_price, 2)
        st.subheader("The predicted price in Lakhs is")
        st.subheader(final_price)


    
