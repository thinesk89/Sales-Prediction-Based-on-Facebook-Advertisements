import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Get the Dataset
from google.colab import files
uploaded = files.upload()

#Read the dataset
df = pd.read_csv('KAG_conversion_data.csv')

#Take a quick look at the dataset
df.head()
df.isnull().sum() #Check for null values in dataset

df.info() #checking for Null values
df.describe() #describing the dataset

print('Total Advertisements from Facebook: {}'.format(df.shape[0]))

print('Number of Ads with ZERO clicks: {}'.format(len(df.loc[df['Clicks'] == 0])))
print('Number of FREE Ads to Facebook: {}'.format(len(df.loc[df['Spent'] == 0])))
print('Number of Ads with ZERO product enquiries: {}'.format(len(df.loc[df['Total_Conversion'] == 0])))
print('Number of Ads with ZERO product purchases: {}'.format(len(df.loc[df['Approved_Conversion'] == 0])))

sns.pairplot(df, corner=True, hue='gender')
plt.suptitle("Numerical features vs. GENDER", x=0.5 ,y=0.95, size=18, weight='bold')

categories = ['ad_id', 'fb_campaign_id','age', 'gender', 'xyz_campaign_id', 'interest']
df[categories] = df[categories].astype('category')
# count plot on single categorical variable 
sns.countplot(x ='xyz_campaign_id', data = df) 
# Show the plot
# displaying the title
plt.title(label="Campaign ID",
          fontsize=20,
          color="red") 
plt.show()

#Approved_Conversion
# Creating bar plot
plt.bar(df["xyz_campaign_id"].astype(str), df["Approved_Conversion"])
plt.ylabel("Approved Conversion of Ads")
plt.xlabel("Campaign")
plt.title("Barplot of Approved Conversion vs Campaign",color = "green")
plt.show()

fig=plt.figure(figsize=(12,7))

sns.countplot(data=df, x='age', hue='gender')
plt.ylabel("Frequency")
plt.xlabel("Age Range")
plt.title('Distribution of Age via Gender', color = "blue")
plt.grid(axis='y')

sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
sns.barplot(x=df["xyz_campaign_id"], y=df["Approved_Conversion"], hue=df["age"], data=tips)
plt.ylabel("Approved Conversion of Ads")
plt.xlabel("Campaign via ID")

fig=plt.figure(figsize=(15,7))

sns.countplot(data=df, x='interest', hue='gender')
plt.title('Interest topic distribution by GENDER')
plt.grid(axis='y')
plt.ylabel("Frequency")
plt.xlabel("Interest Generated")

x=df["interest"]
y=df["Approved_Conversion"]
plt.scatter(x,y,c="red")
plt.xlabel("Interest Generated")
plt.ylabel("Approved Conversion of Ads")
plt.show()

lyx = sns.FacetGrid(df, col="gender", hue = "gender", palette = 'Set1')
lyx.map(plt.scatter, "interest", "Approved_Conversion", alpha=.4)
lyx.add_legend()

tk = sns.FacetGrid(df, col="age", hue ="age", palette = 'Set1')
tk.map(plt.scatter, "interest", "Approved_Conversion", alpha=.4)
tk.add_legend()

plt.scatter(df["Spent"], df["Approved_Conversion"])
plt.title("Money Spent vs. Approved Conversion")
plt.xlabel("Money Spent")
plt.ylabel("Approved Conversion")
plt.show()

dkn = sns.FacetGrid(df, col="gender", hue = "gender", palette = 'Set1')
dkn.map(plt.scatter, "Spent", "Approved_Conversion", alpha=.4)
dkn.add_legend()

c = sns.FacetGrid(df, col="age", hue = "age", palette = 'Set1')
c.map(plt.scatter, "Spent", "Approved_Conversion", alpha=.4)
c.add_legend()

x=df["Impressions"]
y=df["Approved_Conversion"]
plt.scatter(x,y, c ="orange")
plt.title("Aproved Ads Conversion vs. No. of Impressions")
plt.ylabel("Approved Ads Conversion")
plt.show()

h = sns.FacetGrid(df, col="age", hue = "age", palette = 'Set1')
h.map(plt.scatter, "Clicks", "Approved_Conversion", alpha=.4)
h.add_legend()

df["xyz_campaign_id"].unique()
df["xyz_campaign_id"].replace({916:"FB Campaign 1",936:"FB Campaign 2",1178:"FB Campaign 3"}, inplace=True)

# Count of plot on single categorical variable 
sns.countplot(x ='xyz_campaign_id', data = df) 
plt.ylabel("Frequency")
plt.xlabel("Campaign Ads")
# Show the plot 
plt.show()

#encoding gender
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(df["gender"])
df["gender"]=encoder.transform(df["gender"])
print(df["gender"])

#encoding age
encoder.fit(df["age"])
df["age"]=encoder.transform(df["age"])
print(df["age"])

x=np.array(df.drop(labels=["Approved_Conversion","Total_Conversion"], axis=1))
y=np.array(df["Total_Conversion"])

y=y.reshape(len(y),1)
y

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x = sc_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfr.fit(x_train, y_train)

y_predict=rfr.predict(x_test)
y_predict=np.round(y_predict)
y_predict

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
mae=mean_absolute_error(y_test, y_predict)
mse=mean_squared_error(y_test, y_predict)
rmse=np.sqrt(mse)
r2_score=r2_score(y_test, y_predict)
mae
r2_score

from scipy import stats
x=df["Impressions"]
y=df["Spent"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

predicted = myfunc(10)

print(predicted)

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

print(r)

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x=df["Impressions"]
y=df["Spent"]

plt.scatter(x, y)
plt.show()

#Split into Train/Test

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

plt.scatter(train_x, train_y)
plt.show()

from sklearn.metrics import r2_score

r2model = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, r2model(train_x))

print(r2)

print(r2model(1500))

from sklearn import linear_model

t = df[['Impressions', 'Clicks', 'Spent', 'Total_Conversion']]
k = df['Approved_Conversion']

regr = linear_model.LinearRegression()
regr.fit(t, k)

#predict the Approved Conversion where the Impressions are 20,000, Clicks are 5, Spent is 5, Total_Conversion is 3:
predictedAC = regr.predict([[20000,5,5,3]])
print(predictedAC) #Predicted Approved Conversion based on multiple linear regression