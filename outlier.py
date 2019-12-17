# remove outliers from the EEG data
from pandas import read_csv
from numpy import mean
from numpy import std
from numpy import delete
from numpy import savetxt
import numpy as np
from sklearn.preprocessing import normalize

# load the dataset.
# data = read_csv('./EEGEYE_raw.csv', header=None)_

data = read_csv('./EEGEYE.csv', header=None)

values = data.values

for i in range(2,values.shape[1] - 1,1):
    print("Just for checking " + str(values.shape[1]))
	# calculate column mean and standard deviation
    # list1 = list(map(int,values[2:,i]))
    data_mean, data_std = mean(values[:,i]), std(values[:,i])
	# define outlier bounds
    cut_off = data_std * 4
    lower, upper = data_mean - cut_off, data_mean + cut_off
	# remove too small
    too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]
    values = delete(values, too_small, 0)
    print('>deleted %d rows' % len(too_small))
    # remove too large
    too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]
    values = delete(values, too_large, 0)
    print('>deleted %d rows' % len(too_large))
# save the results to a new file
print(values[0,2])
values = np.asarray(values,dtype=np.int64,order='C')
print(values[0,2])
savetxt('EEG_no_outliers.csv', values, delimiter=',',fmt='%0.0f')

#Now comes the normalise step

data = read_csv('./EEG_no_outliers.csv')
new_values = np.zeros((values.shape[0],values.shape[1]))

for i in range(2,values.shape[1],1):
    column = values[:,i]
    max_value = float(np.max(column))
    min_value = float(np.min(column))


    for j in range(values.shape[0]):
        # print(values[j,i])
        #Normalisation step
        print((float(values[j,i]) - min_value))
        print(max_value-min_value)
        new_values[j,i] = float((float(values[j,i]) - min_value)/(max_value-min_value))
        print(new_values[j,i])

savetxt('EEG_norm.csv', new_values, delimiter=',',fmt='%0.3f')



