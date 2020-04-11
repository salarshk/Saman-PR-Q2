import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
df_MT=pd.read_csv('MT_cleaned.csv')
df_VT=pd.read_csv('VT_cleaned.csv')
df_MT.dtypes
por1=sum(df_MT.driver_gender=='M')/len(df_MT.driver_gender)

por2=sum(df_MT.violation=="Speeding")/len(df_MT.violation)

df_MT.dropna(subset=['stop_time'], inplace=True)
df_MT.dropna(subset=['stop_date'], inplace=True)
df_MT["stop_date"]=pd.to_datetime(df_MT["stop_date"])
df_MT["stop_time"]=pd.to_datetime(df_MT["stop_time"])
k=0;
year=np.zeros([len(df_MT["stop_date"]),1])
for i in df_MT["stop_date"]:
    year[k]=i.year
    k=k+1
k=0
hour=np.zeros([len(df_MT["stop_time"]),1])
minute=np.zeros([len(df_MT["stop_time"]),1])
for i in df_MT["stop_time"]:
    hour[k]=i.hour
    minute[k]=i.minute
    k=k+1
df_MT["year"]=year
df_MT["hour"]=hour
df_MT["minute"]=minute
df_VT.dropna(subset=['stop_time'], inplace=True)
df_VT.dropna(subset=['stop_date'], inplace=True)
df_VT["stop_date"]=pd.to_datetime(df_VT["stop_date"])
df_VT["stop_time"]=pd.to_datetime(df_VT["stop_time"])
k=0;
year=np.zeros([len(df_VT["stop_date"]),1])
for i in df_VT["stop_date"]:
    year[k]=i.year
    k=k+1
k=0
hour=np.zeros([len(df_VT["stop_time"]),1])
minute=np.zeros([len(df_VT["stop_time"]),1])
for i in df_VT["stop_time"]:
    hour[k]=i.hour
    minute[k]=i.minute
    k=k+1
df_VT["year"]=year
df_VT["hour"]=hour
df_VT["minute"]=minute
un_years=df_MT["year"].unique()
num_years=np.zeros([len(un_years)])
num_years_man=np.zeros([len(un_years)])
chi_table=np.zeros([2,2])
chi_table[0,0]=len(df_MT.loc[(df_MT["driver_gender"]=="M") & (df_MT["is_arrested"]==True)])
chi_table[0,1]=len(df_MT.loc[(df_MT["driver_gender"]=="M") & (df_MT["out_of_state"]==True)])
chi_table[1,0]=len(df_MT.loc[(df_MT["driver_gender"]=="F") & (df_MT["is_arrested"]==True)])
chi_table[1,1]=len(df_MT.loc[(df_MT["driver_gender"]=="F") & (df_MT["out_of_state"]==True)])
chi_table=pd.DataFrame(chi_table,index=["Male","Female"],columns=["arrested","out_of_state"])
stat, p, dof, expected = chi2_contingency(chi_table)
k=0
for i in range(len(un_years)):
    num_years[i]=sum(df_MT["year"]==un_years[i])
    year_temp=df_MT[df_MT["year"]==un_years[i]]["vehicle_year"]
    year_temp=year_temp[year_temp!="UNK"]
    year_temp=year_temp[year_temp!="NON-"]  
    num_years_man[i]=np.mean(year_temp.dropna().astype(float))
num_years_man=np.round(num_years_man)
X=un_years
y=num_years
print("The proportion of traffic stops in MT involving male drivers is:")
print(por1)
print("Chi-Squared traffic stop arrest test statistic is:")
print(stat)
print("The proportion of traffic stops in MT involving speeding violations is:")
print(por2)
print("The average manufacture year of vehicles stopped in MT between(2009-2016) are:")
print(num_years_man)
Dif_MT=sum(df_MT.hour==max(df_MT.hour))-sum((df_MT.hour==min(df_MT.hour)))
Dif_VT=sum(df_VT.hour==max(df_MT.hour))-sum((df_VT.hour==min(df_MT.hour)))
print("The difference in the total number between min and max hours (MT):")
print(Dif_MT)
print("The difference in the total number between min and max hours (VT):")
print(Dif_VT)
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
#print(est2.summary())
Y_pred = est2.predict(X2)
plt.title("the number of traffic stops by the year and its regression (2009-2016)")
plt.scatter(X, y)
plt.plot(X, Y_pred, color='red')
plt.show()
X_Corr=un_years[1:]
y_Corr=num_years[1:]
X2_Corr = sm.add_constant(X_Corr)
est_Corr = sm.OLS(y_Corr, X2_Corr)
est2_Corr = est_Corr.fit()
#print(est2_Corr.summary())
Y_pred_Corr = est2_Corr.predict(X2_Corr)
print("P-value of linear regression is in the second table as p>|t|:")
print(est2_Corr.summary())
plt.title("the number of traffic stops by the year and its regression (2010-2016)")
plt.scatter(X_Corr, y_Corr)
plt.plot(X_Corr, Y_pred_Corr, color='red')
plt.show()
plt.title("average manufacture year of vehicles stopped by years")
plt.plot(un_years, num_years_man, color='red')
plt.show()

#mymodel = np.poly1d(np.polyfit(X, y,1))
#myline = np.linspace(2009, 2020, 10000)
#plt.scatter(X, y)
#plt.plot(myline, mymodel(myline))
#plt.show()
