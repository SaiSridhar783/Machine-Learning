## Problem Description
Imagine you are working in a fashion e-commerce startup. You have to predict the yearly amount spent by your customers, for your team to devise business strategies.

## Features
You have the customer's data which include :
- Time Spent on Website
- Duration of Membership 
- Time Spent on App 
- Session Duration 
- Yearly amount spent by them.


# HELPER FUNCTIONS

To read a csv file and convert into numpy array, you can use genfromtxt of the numpy package.
For Example:
```
train_data = np.genfromtxt(train_X_file_path, dtype=np.float64, delimiter=',', skip_header=1)
```
You can use the python csv module for writing data to csv files.
Refer to https://docs.python.org/2/library/csv.html.
For Example:
```
with open('sample_data.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
    writer.writerows(data)
```
