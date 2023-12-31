import pandas as pd

weather = pd.read_csv("data.csv", index_col="DATE")
print(weather)

# 1. dealing with missing (null) values

# shows percentage of null values in each column
print(weather.apply(pd.isnull).sum()/weather.shape[0]) 

# prepare a smaller set of data including only the core weather parameters
core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy() 
print(core_weather)

# shows percentage of null values in each column
print(core_weather.apply(pd.isnull).sum()/core_weather.shape[0]) 

# check the values and update the data set
print(core_weather["SNOW"].value_counts())
del core_weather["SNOW"]
print(core_weather["SNWD"].value_counts())
del core_weather["SNWD"]

# select rows where precipitation is null
print(core_weather[pd.isnull(core_weather["PRCP"])])

# replacing missing values
# solution: most of the values are zeros
print(core_weather["PRCP"])
print(core_weather["PRCP"].value_counts())
# so let's replace the missing values with zeros
core_weather["PRCP"] = core_weather["PRCP"].fillna(0)
print(core_weather["PRCP"])

# select rows where temperature is null
print(core_weather[pd.isnull(core_weather["TMAX"])])
print(core_weather[pd.isnull(core_weather["TMIN"])])

# replace the null value with the previous value which was not null
core_weather = core_weather.fillna(method="ffill")

# check the percentage of null values in each column one more time 
print(core_weather.apply(pd.isnull).sum()/core_weather.shape[0]) 

# 2. verifying the data types (the numeric types are needed!)

print(core_weather.index)
core_weather.index = pd.to_datetime(core_weather.index)

# 3. remove the "9999" which indicates the missing data

core_weather.apply(lambda x: (x==9999).sum())

# 4. the weather data analysis

core_weather[["TMAX", "TMIN"]].plot()
core_weather[["PRCP"]].plot()

# count the days in each year where there is data
print(core_weather.index.year.value_counts().sort_index())

# count the sum of rain per each year
print(core_weather.groupby(core_weather.index.year).sum()["PRCP"])

# 5. training the simple machine learning model

# the aim is to predict the tomorrow's temperature

# shifting the columns
# the target for  each row is the temperature of the following day
core_weather["target"] = core_weather.shift(-1)["TMAX"]
print(core_weather)

# delete the last row so there is no null values
core_weather = core_weather.iloc[:-1,:].copy()
print(core_weather)

# machine learning model
from sklearn.linear_model import Ridge
reg = Ridge(alpha=.1)

predictors = ["PRCP", "TMAX", "TMIN"]

# split the data into train and test sets
train = core_weather.loc[:"2020-12-31"]
test = core_weather.loc["2021-01-01":]

# fitting the model
reg.fit(train[predictors], train["target"])
predictions = reg.predict(test[predictors])

# calculate the mean_absolute_error
# which is the avarage of the difference between the actual values and the predictions
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["target"], predictions)
print(mean_absolute_error)

# 6. evaluating the model 

# comparing the true and predicted values
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", 'predictions']
print(combined)
print(combined.plot())

# the coefficients
print(reg.coef_)

# 7. predictions function

def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", 'predictions']
    return error, combined

# 8. adding rolling means

core_weather["month_max"] = core_weather['TMAX'].rolling(30).mean()
print(core_weather)
core_weather["month_day_max"] = core_weather["month_max"] / core_weather["TMAX"]
core_weather["max_min"] = core_weather["TMAX"] / core_weather["TMIN"]
predictors = ["PRCP", "TMAX", "TMIN", "month_max", "month_day_max", "max_min"]

# remove the first 30 null values
core_weather = core_weather.iloc[30:,:].copy()

# create predictions with new predictors
error, combined = create_predictions(predictors, core_weather, reg)
print(error)

# the averages of TMAX for each month (all the years) before particular observation (no future data)
core_weather["monthly_avg"] = core_weather["TMAX"].groupby(core_weather.index.month).apply(lambda x: x.expanding(1).mean())
print(core_weather)

# the avarage of TMAX of previous 30 days for particular observation
core_weather["day_of_year_avg"] = core_weather["TMAX"].groupby(core_weather.index.day_of_year).apply(lambda x: x.expanding(1).mean())
predictors = ["PRCP", "TMAX", "TMIN", "month_max", "month_day_max", "max_min", "day_of_year_avg", "monthly_avg"]

# create predictions with new predictors
error, combined = create_predictions(predictors, core_weather, reg)
print(error)

# 9. running model diagnostics 

# the coefficients
print(reg.coef_)

# corelations between target and predictors
print(core_weather.corr()["target"])

# differences between actual and predicted values
combined["diff"] = (combined["actual"] - combined["predictions"]).abs()
print(combined.sort_values("diff", ascending=False).head())
