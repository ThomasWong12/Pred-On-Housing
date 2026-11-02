import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

test = pd.read_csv("/Users/thomaswong/Downloads/regression-for-house-prices (1)/ames/test.csv")
train = pd.read_csv("/Users/thomaswong/Downloads/regression-for-house-prices (1)/ames/train.csv")
print(train.columns)

# Im going to try to build a multi demosonial linear regression model 
# Use the training set of data which will be train 
print(train['SalePrice'])
# Its important to remeber that for many of the vaibles that im going to need to add something such as a one hot encoders 
# or an ordinal value to fit it to a linear regressionb model that im going to do 
# That being said i got about 10 hours to hammer out this project so lets build somethig great!!!

# Lets start with something that take 2 number values which should not need to take much transforming 
# Then lets take one thing in that i will transform usig ohe.

X = train[['LotArea','OverallQual','OverallCond','HeatingQC','LotShape']]
y = train['SalePrice']
# Two ord values in the overalls 
# Lets we can just plug in 
# lot shape need to use an unordered dummy value attacheted to it
# an idea could be to squar all the submission so we could have a better model 

X['LotArea^2'] = X['LotArea']**2
X['OverallQual^2'] = X['OverallQual']**2
X['OverallCond^2'] = X['OverallCond']**2

# Just checking if any of the values from my data have gone missing 
print(train.columns[train.isna().sum()>0])
# There not so we should be fine
# This part should be for the dummy values that dont matter 
onehot = OneHotEncoder(drop = 'first')
onehot.fit_transform(train[['LotShape']]).todense()
onehot.get_feature_names_out()
# This part should be for the values that are actually important to be ordered
cats = [['Po','Fa','TA','Gd','Ex']]
ordinal = OrdinalEncoder( categories = cats)
ordinal.fit_transform(train[['HeatingQC']])
# We should start putting everything together, this will require us to use both a transformer and the a piple as this is a muli model
thing = make_column_transformer(
    (onehot, ['LotShape']),
    (ordinal, ['HeatingQC']),
    remainder = 'passthrough'
)
XT = thing.fit_transform(X)
model = LinearRegression().fit(XT,y)
pipeline = Pipeline([
    ('encode', thing), 
    ('model', LinearRegression())
    ])
fitting = pipeline.fit(X, y)
# Lets try to make some predictions about the data that is given to us.
# Belove is the formatting of our data 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=42
    )
pipeline.fit(X_train, y_train)
val = pipeline.predict(X_valid)
