

## Pandas
Provides **data structure and data analysis tools** like Data Frame that is basically a table but better and easy to manipulate than other libraries.

#### How to Manipulate a Data Frame
  * Create
    + pd.DataFrame([fhand or npArray or list])
    + pd.DataFrame(fhand, index=['equal to n rows'], columns=['equal to n columns'])
  * Read
    + columns : dot notation | rows : column + ['row_name'] -> df.Q1['2019']
    + head(), tail(), type(), .index, .columns, shape (rows,columns)
  * Get
    + df.index.get_loc('row_name') / df.columns.get_loc('col_name') : return its position.
    + cities.column.unique() : return unique values.
    + df.iloc[<row number>,<column number>] | df.loc['row label','column label']
      - To select an entire row using iLoc: df.iloc[1,:]
      - To select an entire column using iloc: quarterly_sales.iloc[:,1]
  * Update
    + nwcity = cities.rename(columns={'old':'new'})
    + nwcity = cities.drop(columns/index=['name'])

##### Pandas Pivot Tables
Aggregate and compare large datasets.
~~~
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
df_piv = tips.pivot_table('tip', index='sex', columns='time')

>>> time       Lunch    Dinner
>>> sex                       
>>> Male    2.882121  3.144839
>>> Female  2.582857  3.002115
~~~

##### Split Columns & Extract Data Using Delimiters
Let’s say that your first column has values like in a CSV document that uses semi-column spacer.
~~~
## Column A
## 322;435;423
## 111;2443;23556
## 222
## 111;354
To split columns using spacers in Pandas, use the str.split function.

newdf = df.iloc[:,1].str.split(";", expand = True)
~~~

##### to keep things simple, we use only numerical predictors
~~~
columns = fhand.drop(['Price'],axis=1)
x = columns.select_dtypes(exclude=['object'])
~~~


##### simple get of caterogirical column names
~~~
# get all the categorical column names
s = (train_x.dtypes == 'object')
object_cols = list(s[s].index)
~~~

## Numpy
NumPy, short for Numerical Python, provides efficient storage and manipulation of numerical arrays that you can use for advanced calculations instead of regular Python lists.

The arrays, like the lists, are used to store multiple values in one single variable. The main difference is that you can perform **calculations over entire arrays**.

#### How to Manipulate a Numpy Array
Deben ser Arrays iguales a row x column, es decir, numpy crea arrays completos y sin excedentes.
  * Create
    + np.zeros(n_zeros, dtype=int)
    + matrix : np.full((n_col,n_items_per_arr),value)
    + 1D array: np.array([8,9,13])
    + 1D array: np.arange(n_range_of_items) / np.arange(n_start,n_end,n_increment)
    + 2D array: np.arange(n_range_of_items).reshape(n_col,n_items_per_arr)
    + 3D array: np.arange(n_range_of_items).reshape(n_row,n_col,n_items_per_arr)
    + 1D array of random values between 0 & 1 : np.random.rand(n_col)
  * Read
    + arr.shape : (row,col)
    + arr.ndim : N Dimensions (the array itself is always the first dimension)
    + arr.size : Total number of items

#### NumPy Statistical Functions
As a Data Scientist, you’ll be faced with a number of statistical problems that can be solved using NumPy Functions. Here is a quick overview of the functions that you might be interested in:

  * np.mean() : Calculate the Mean of an array
  * np.median() : Calculate the Median of an array
  * np.max(): Find the highest value
  * np.min(): Find the lowest value
  * np.corrcoef(x,y) : Find Correlation
  * np.std() : Compute Standard deviation
  * np.sqrt() : Square Root
  * np.sum(): Calculate the Sum of an array
  * np.sort(): Sort Data


> Pandas and Numpy information was resume from https://www.jcchouinard.com/python-libraries-for-seo/


## Scikit-learn (sklearn.xxx)
**Regardless of the model or algorithm, the code structure for training and prediction is the same**.

List of packages and use of them.
  <!-- * from sklearn.tree import DecisionTreeRegressor -->
  * from sklearn.ensemble import RandomForestRegressor
  * from sklearn.model_selection import train_test_split
  * from sklearn.metrics import mean_absolute_error (score)
  * from sklearn.impute import SimpleImputer
    + imputer.fit_transform(train_x)
    + imputer.transform(vtion_x)
  * from sklearn.processing import LabelEncoder
    + label_encoder.fit_transform(label_train_x[col])
    + label_encoder.transform(label_vtion_x[col])
  * from sklearn.processing impor OneHotEncoder
    + OneHotEncoder(handle_unknown='ignore',sparse=False)














































#
