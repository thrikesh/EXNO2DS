# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
## NAME: THRIKESWAR
## REGISTER NUMBER: 212222230162
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```

![Screenshot 2024-03-09 113810](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/a4c59ae2-f1ac-45d7-8558-829f53ae0c99)
```
dt.info()
```
![Screenshot 2024-03-09 111848](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/51c81a95-0ce4-43e9-869b-49ad5f6b5a72)
```
dt.shape
```
![Screenshot 2024-03-09 111927](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/6ba66330-6491-4e5d-9ce7-dc825acfab99)

```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
![Screenshot 2024-03-09 112037](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/7f84a707-b51b-474b-a9c8-e193aab04b63)
```
dt.nunique()
```
![Screenshot 2024-03-09 112112](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/cde13e60-8796-4976-b6ba-c4440c2d9eb3)
```
dt["Survived"].value_counts()
```
![Screenshot 2024-03-09 112211](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/b84327bd-7782-4cf3-8fe7-e18d68ea92c7)
```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![Screenshot 2024-03-09 112247](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/b641e117-f0da-4d64-a886-8964a4e91122)
```
sns.countplot(data=dt,x="Survived")
```
![Screenshot 2024-03-09 112334](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/a59d025d-9c1d-47c2-aaf2-874ef46924ad)
```
dt
```
![Screenshot 2024-03-09 111446](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/f4d4558c-b01e-45aa-b0a9-15e5c6a8177f)
```
dt.Pclass.unique()
```
![Screenshot 2024-03-09 112614](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/406e9fde-ecd5-408a-89de-aad0e3da762a)
```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![Screenshot 2024-03-09 112728](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/f0c27766-b897-4993-959d-6c1b7f24f751)

```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![Screenshot 2024-03-09 112816](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/37c1872f-3165-4c94-8ab4-13e5b8eb0b17)
```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
![Screenshot 2024-03-09 112845](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/64b8ea04-9d73-4022-a273-d69d1b8156ee)
```
dt.boxplot(column="Age",by="Survived")
```
![Screenshot 2024-03-09 112906](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/a38de3af-ff5e-4cc0-b895-528e04069548)
```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![Screenshot 2024-03-09 113012](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/03e6b815-fb07-431b-b8dc-97a5ac4ea7e3)
```
sns.jointplot(x="Age",y="Fare",data=dt)
```
![Screenshot 2024-03-09 113049](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/705d311f-7324-4220-9a5c-87da77f82b73)
```
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass",y="Age",hue="Gender",data=dt)
```
![Screenshot 2024-03-09 113142](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/5e822a5e-a656-4eed-8af4-e060aa2fb47c)

```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![Screenshot 2024-03-09 113229](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/b26290b1-f119-4ec1-9f7b-9c24b176d005)
```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![Screenshot 2024-03-09 113316](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/779b325d-97d0-49ad-ba59-d9ceedd0dd1b)
```
sns.pairplot(dt)
```
![Screenshot 2024-03-09 113415](https://github.com/Anusharonselva/EXNO2DS/assets/119405600/b5ca2152-7f38-47a0-8cef-121951f80d18)


# RESULT
   Thus, the Exploratory Data Analysis on the given data set was performed successfully.
