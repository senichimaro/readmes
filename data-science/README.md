# Commands Cheatsheet
sales = pd.read_csv(fhand)

sales.shape : (rows, columns) info about how many rows and columns data variable has.

sales.info() : info about Dtype, comlumn names, etc...

sales.describe() : statistical properties

sales.head() : first 5 rows

sales['Age_Group'].value_counts() : total category values for each category in the column
~~~
>>> tall 150
>>> smal 95
>>> bigS 43
~~~
sales.corr() : correlation analisys

sales.loc[criteria] : filter
~~~
# get all the sales made in kentucky
sales.loc[sales['State'] == 'Kentucky']
# 'Revenue' cnew column name
sales.loc[sales['Country'] == 'France', 'Revenue'] *= 1.1

~~~





























































































#
