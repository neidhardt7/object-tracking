import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("slipvel.xlsx", sheet_name ="Sheet2", header = 1)

data = df.values
print(data)

x = data[:,0]
print(x)

y = data[:,1]
print(y)

y_error = data [:, 2]
print(y_error)

plt.plot(x,y,color= "blue", marker = "o")
plt.errorbar(x,y,yerr= y_error, fmt="-",color= "red",capsize = 2)
plt.xscale("log")
plt.xlabel('relative sds concentration')
plt.ylabel('velocity(mm/sec)')


#plt.savefig('slipvel')
