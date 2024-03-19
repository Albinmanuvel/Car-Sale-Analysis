import os
os.getcwd()
os.chdir("F:/hadoop")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
D1=pd.read_csv("CarPrice.csv")
D1.dtypes
D1.isnull().sum()

fd=D1
fd = pd.DataFrame(fd)
fd["aspiration"] = fd["aspiration"].fillna(method='ffill')
fd["cylindernumber"] = fd["cylindernumber"].fillna(method='ffill')
fd["horsepower"] = fd["horsepower"].fillna(method='ffill')

fd.isnull().sum()

fd.highwaympg.unique()
fd.citympg.unique()
fd.peakrpm.unique()
fd.compressionratio.unique()
fd.drivewheel.unique()
fd.enginesize.unique()
fd.enginelocation.unique()
fd.horsepower.unique()
fd.compressionratio.unique()
fd.stroke.unique()
fd.boreratio.unique()
fd.fuelsystem.unique()
fd.curbweight.unique()
fd.carheight.unique()
fd.compressionratio.unique()


import pingouin as pp

pp.welch_anova(fd,dv="symboling",between="price")
pp.welch_anova(fd,dv="price",between="carbody")      
pp.welch_anova(fd,dv="price",between="enginelocation") 

pp.welch_anova(fd,dv="price",between="doornumber")     #remove
pp.welch_anova(fd,dv="price",between="fueltype")       #remove

pp.welch_anova(fd,dv="price",between="aspiration")
pp.welch_anova(fd,dv="price",between="drivewheel")
pp.welch_anova(fd,dv="price",between="enginetype")
pp.welch_anova(fd,dv="price",between="cylindernumber")
pp.welch_anova(fd,dv="price",between="fuelsystem")



import scipy.stats as stats
stats.pearsonr(fd["horsepower"],fd["price"])
stats.pearsonr(fd["highwaympg"],fd["price"])
stats.pearsonr(fd["citympg"],fd["price"])
stats.pearsonr(fd["citympg"],fd["price"])

stats.pearsonr(fd["peakrpm"],fd["price"])            #remove
stats.pearsonr(fd["compressionratio"],fd["price"])   #remove
stats.pearsonr(fd["stroke"],fd["price"])             #remove
stats.pearsonr(fd["carheight"],fd["price"])          #remove

stats.pearsonr(fd["boreratio"],fd["price"])
stats.pearsonr(fd["enginesize"],fd["price"])
stats.pearsonr(fd["curbweight"],fd["price"])
stats.pearsonr(fd["wheelbase"],fd["price"])
stats.pearsonr(fd["carlength"],fd["price"])
stats.pearsonr(fd["carwidth"],fd["price"])


fdata=fd.drop(["symboling","doornumber","fueltype","peakrpm","compressionratio","stroke","carheight","car_ID","CarName"], axis=1)

from sklearn.model_selection import train_test_split
x = fdata.drop('price', axis=1)
y = fdata['price']
x=pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)                         

import xgboost
from xgboost import XGBRFRegressor
model=XGBRFRegressor()
model.fit(x_train,y_train)
y2_pred=model.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y2_pred)                        

from sklearn.tree import DecisionTreeRegressor
m=DecisionTreeRegressor()
m.fit(x_train,y_train)
y3_pred=m.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test, y3_pred)                        

from sklearn.ensemble import RandomForestRegressor
m1=RandomForestRegressor()
m1.fit(x_train,y_train)
y4_pred=m1.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test, y4_pred)