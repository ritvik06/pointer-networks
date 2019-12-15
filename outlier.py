# remove outliers from the EEG data
from pandas import read_csv
from numpy import mean
from numpy import std
from numpy import delete
from numpy import savetxt
# load the dataset.
# data = read_csv('./EEGEYE_raw.csv', header=None)
with open("EEGEYE_raw.csv",'r') as f:
    with open("EEGEYE.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)

# step over each EEG column

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
savetxt('EEG_no_outliers.csv', values, delimiter=',')