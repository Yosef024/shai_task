import pandas as pd
import numpy as np
import seaborn as sn
from collections import Counter
dataset = pd.read_csv("Salaries.csv")
dataset = dataset[dataset['TotalPay'] > 1000]
# step 1
NumRows, NumCol = dataset.shape
print("columns", NumCol)
print("rows", NumRows)

type = dataset.dtypes
print("type", type)

missing = dataset.isnull()
print("missing values is \n", missing.sum())

# step 2
meanSalary = dataset['TotalPay'].mean()
medianSalary = dataset['TotalPay'].median()
modeSalary = dataset['TotalPay'].mode()
minSalary = dataset['TotalPay'].min()
maxSalary = dataset['TotalPay'].max()
rangeSalary = maxSalary-minSalary
stdSalary = dataset['TotalPay'].std()
print("mean=", meanSalary)
print("median=", medianSalary)
print("mode=", modeSalary)
print("min=", minSalary)
print("max=", maxSalary)
print("range=", rangeSalary)
print("standard deviation=", stdSalary)

# step 3
# I will use mean ,but we can use mean or median because it's a suitable values unlike mode 0
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
missingColumns = dataset.iloc[:, 3:7].values
imputer.fit(missingColumns)
dataset.iloc[:, 3:7] = imputer.transform(missingColumns)

# step 4
import matplotlib.pyplot as plt
plt.hist(dataset['TotalPay'], bins=50, color='purple', edgecolor='black')
plt.xlabel('salary')
plt.ylabel('employees number')
plt.title('distribution of employees salaries')
plt.show()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
agencyColumn = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
agency = dataset['Year'].values.reshape(-1, 1)
agency = np.array(agency)
values,agencyCounts=np.unique(agency,return_counts=True)
print(values)
print(agencyCounts)
plt.pie(agencyCounts,labels=values)
plt.title('employee distribution in different years')
plt.show()

agencyColumn = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
agency = dataset['JobTitle'].values.reshape(-1, 1)
agency = np.array(agency[:50,:])
values,agencyCounts=np.unique(agency,return_counts=True)
print(values)
print(agencyCounts)
plt.pie(agencyCounts,labels=values)
plt.title('employee distribution in different years')
plt.show()

# step 5
overtpay = dataset.groupby(['OvertimePay'])['TotalPay'].mean().reset_index()
basepay=dataset.groupby(['BasePay'])['TotalPay'].mean().reset_index()
plt.scatter(overtpay.iloc[:, 0].tolist(),overtpay.iloc[:, 1].tolist(),color='red')
plt.plot(basepay.iloc[:, 0].tolist(),basepay.iloc[:, 1].tolist(),color='blue',alpha=0.5)
plt.title('overtime pay vs base pay salary avg')
plt.xlabel('overtime pay and base pay')
plt.ylabel('salary')
plt.show()



#step 6
correlation=dataset['TotalPay'].corr(dataset['BasePay'])
sn.scatterplot(x='TotalPay',y='BasePay',data=dataset)
plt.title('TotalPay vs BasePay')
plt.xlabel('TotalPay')
plt.ylabel('BasePay')
plt.show()

correlation=dataset['TotalPay'].corr(dataset['OtherPay'])
sn.scatterplot(x='TotalPay',y='OtherPay',data=dataset)
plt.title('TotalPay vs OtherPay')
plt.xlabel('TotalPay')
plt.ylabel('OtherPay')
plt.show()