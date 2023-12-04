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
