from pandas import read_csv
from numpy import mean
from numpy import std
from numpy import delete
from numpy import savetxt
import numpy as np
from sklearn.preprocessing import normalize

data = read_csv('./EEGEYE.csv', header=None)

values = data.values

is_CP = []
boundaries = []
gap = []
values_down = np.zeros((values.shape[0]/10 + 1,16))

print(values_down.shape[0])

def is_toggle(x):
    if (x==1):
        return True
    
    return False

for i in range(0,values.shape[0]-1,1):
    if (is_toggle(values[i][0])==1):
        is_CP.append(i)

for i in range(len(is_CP)-1):
    if(i%2==0):
        tup = (is_CP[i],is_CP[i+1],is_CP[i+1]-is_CP[i])
        boundaries.append(tup)
        gap.append(is_CP[i+1]-is_CP[i])

print("Mean " + str(mean(gap)))    #610
print("Max " + str(max(gap)))      #2401
print("Min " + str(min(gap)))      #27

#Down Sampling of data, 15000 data points for 117 seconds.
# Down sample 10 data points 

for i in range(0,values.shape[0],10):
    if 1 in values[i:i+10,0]:
        values_down[i//10][0]=1
    else:
        values_down[i//10][0]=0

    values_down[i//10][1] = 0 #Leaving this column as random, no need of it
    for j in range(2,16):
        values_down[i//10][j] = mean(values[i:i+10,j]) 

savetxt('EEG_down.csv', values_down, delimiter=',',fmt='%0.3f')








