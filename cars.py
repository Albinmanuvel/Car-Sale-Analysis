import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
D1=pd.read_csv("CarPrice.csv")
D1.dtypes
D1.isnull().sum()

fd=D1
fd = pd.DataFrame(fd)
fd["aspiration"] = fd["aspiration"].ffill()
fd["cylindernumber"] = fd["cylindernumber"].ffill()
fd["horsepower"] = fd["horsepower"].ffill()

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

def welch(df, dv, between):
    # some categories have a single car (e.g. dohcv engines), which Welch ANOVA cannot handle
    try:
        print(pp.welch_anova(df, dv=dv, between=between))
    except ValueError as e:
        print(between, 'skipped:', e)

welch(fd,"price","symboling")
pp.welch_anova(fd,dv="price",between="carbody")      
pp.welch_anova(fd,dv="price",between="enginelocation") 

welch(fd,"price","doornumber")     #remove
welch(fd,"price","fueltype")       #remove

welch(fd,"price","aspiration")
welch(fd,"price","drivewheel")
welch(fd,"price","enginetype")
welch(fd,"price","cylindernumber")
welch(fd,"price","fuelsystem")



import scipy.stats as stats
print(stats.pearsonr(fd["horsepower"],fd["price"]))
print(stats.pearsonr(fd["highwaympg"],fd["price"]))
print(stats.pearsonr(fd["citympg"],fd["price"]))
print(stats.pearsonr(fd["citympg"],fd["price"]))

print(stats.pearsonr(fd["peakrpm"],fd["price"]))            #remove
print(stats.pearsonr(fd["compressionratio"],fd["price"]))   #remove
print(stats.pearsonr(fd["stroke"],fd["price"]))             #remove
print(stats.pearsonr(fd["carheight"],fd["price"]))          #remove

print(stats.pearsonr(fd["boreratio"],fd["price"]))
print(stats.pearsonr(fd["enginesize"],fd["price"]))
print(stats.pearsonr(fd["curbweight"],fd["price"]))
print(stats.pearsonr(fd["wheelbase"],fd["price"]))
print(stats.pearsonr(fd["carlength"],fd["price"]))
print(stats.pearsonr(fd["carwidth"],fd["price"]))


fdata=fd.drop(["symboling","doornumber","fueltype","peakrpm","compressionratio","stroke","carheight","car_ID","CarName"], axis=1)

from sklearn.model_selection import train_test_split
x = fdata.drop('price', axis=1)
y = fdata['price']
x=pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
from sklearn.metrics import r2_score
print("Linear Regression R2:", round(r2_score(y_test,y_pred),3))
import xgboost
from xgboost import XGBRFRegressor
model=XGBRFRegressor(random_state=42)
model.fit(x_train,y_train)
y2_pred=model.predict(x_test)
from sklearn.metrics import r2_score
print("XGBoost RF R2:", round(r2_score(y_test,y2_pred),3))
from sklearn.tree import DecisionTreeRegressor
m=DecisionTreeRegressor(random_state=42)
m.fit(x_train,y_train)
y3_pred=m.predict(x_test)
from sklearn.metrics import r2_score
print("Decision Tree R2:", round(r2_score(y_test,y3_pred),3))
from sklearn.ensemble import RandomForestRegressor
m1=RandomForestRegressor(random_state=42)
m1.fit(x_train,y_train)
y4_pred=m1.predict(x_test)
from sklearn.metrics import r2_score
print("Random Forest R2:", round(r2_score(y_test,y4_pred),3))