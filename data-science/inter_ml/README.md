# Intermediate Machine Learning
Learn to handle missing values, non-numeric values, data leakage and more.

## Missing Values
Approaches to dealing with missing values. Then you'll compare the effectiveness of these approaches on a real-world dataset.

If you try to build a model using data with **missing values in libraries (including scikit-learn) give an error**. So you'll need to choose one of the strategies below.

### Three Approaches to deal with missing values
  1. Simple Option: Drop Columns with Missing Values
  2. Standard approach: Imputation (A Better Option)
  3. An Extension To Imputation

#### Simple Option: Drop Columns with Missing Values
![drop entire column](drop-column.png)
The model **loses access to a lot of (potentially useful!) information** with this approach. As an extreme example, consider a dataset with 10,000 rows, where one important column is missing a single entry. This approach would drop the column entirely!

#### Standard approach: Imputation (A Better Option)
![input fills in the missing values](imputation.png)
Imputation fills in the missing values with some number. For instance, we can fill in the mean value along each column. **The imputed value won't be exactly right in most cases, but it usually leads to more accurate models** than you would get from dropping the column entirely.

Although it's simple, filling in the mean value generally performs quite well (but this varies by dataset). While statisticians have experimented with more complex ways to determine imputed values (such as regression imputation, for instance), the complex strategies typically give no additional benefit once you plug the results into sophisticated machine learning models.

#### An Extension To Imputation
The model would make **better predictions by considering which values were originally missing**.
In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.
![Considering which values were originally missing](inputation-extension.png)

#### Example
In the example, we will work with the Melbourne Housing dataset. Our model will use information such as the number of rooms and land size to predict home price.
~~~
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_absolute_error
  from sklearn.impute import SimpleImputer

  # load file
  fpath = "mdrent.csv"
  fhand = pd.read_csv(fpath)

  # target & predictor variable
  y = fhand.Price
  # to keep things simple, we use only numerical predictors
  columns = fhand.drop(['Price'],axis=1)
  x = columns.select_dtypes(exclude=['object'])

  # training and validation subsets
  train_x, vtion_x, train_y, vtion_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

  # function for comparing different approaches
  def score_dataset(train_x, vtion_x, train_y, vtion_y):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(train_x, train_y)
    preds = model.predict(vtion_x)
    return mean_absolute_error(vtion_y, preds)

  # ## Approach 1 (Drop Columns with Missing Values)

  # Get names of columns with missing values
  cols_with_missing = [col for col in train_x.columns if train_x[col].isnull().any()]

  # Drop columns in training and validation data
  reduced_train_x = train_x.drop(cols_with_missing, axis=1)
  reduced_vtion_x = vtion_x.drop(cols_with_missing, axis=1)

  print("Approach 1 (Drop Columns with Missing Values)")
  print("Missing Columns:", cols_with_missing)
  print("MAE:",score_dataset(reduced_train_x, reduced_vtion_x, train_y, vtion_y))


  # ## Approach 2 (Imputation or "Cleaning the data without removing values")
  # Empty or non-numeric values are sustituted by the mean value for that column

  # Imputation
  imputer = SimpleImputer()
  # Data values are cleaned but it lost the column names
  imputed_train_x = pd.DataFrame(imputer.fit_transform(train_x))
  imputed_vtion_x = pd.DataFrame(imputer.transform(vtion_x))

  # removed column names are put them back
  imputed_train_x.columns = train_x.columns
  imputed_vtion_x.columns = vtion_x.columns
  print("MAE:", score_dataset(imputed_train_x, imputed_vtion_x, train_y, vtion_y))


  # ## Approach 3 (An Extension to Imputation)
  # impute the missing values, while also keeping track of which values were imputed.



  # make new columns indicating what will be imputed
  for col in cols_with_missing:
    train_x_plus[col + '_was_missing'] = train_x_plus[col].isnull()
    vtion_x_plus[col + '_was_missing'] = vtion_x_plus[col].isnull()

  # imputation
  imputer2 = SimplerImputer()
  imputed_train_x_plus = pd.DataFrame(imputer2.fit_transform(train_x_plus))
  imputed_vtion_x_plus = pd.DataFrame(imputer2.transform(vtion_x_plus))

  # Removed column names put them back
  imputed_train_x_plus.columns = train_x_plus.columns
  imputed_vtion_x_plus.columns = vtion_x_plus.columns

  print("MAE:", score_dataset(imputed_train_x_plus, imputed_vtion_x_plus, train_y, vtion_y))
~~~


### So, why did imputation perform better than dropping the columns?
The training data has 10864 rows and 12 columns, where three columns contain missing data. For each column, less than **half of the entries are missing. Thus, dropping the columns removes a lot of useful information**, and so it makes sense that imputation would perform better.
~~~
# Shape of training data (num_rows, num_columns)
print("\nShape of training data (num_rows, num_columns):", train_x.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (train_x.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
~~~


# Categorical Variables
The data is categorical because responses fall into a fixed set of categories.

**You will get an error if you try to plug these variables into most machine learning models in Python without preprocessing them first**.

We'll compare three approaches that you can use to prepare your categorical data.

### Drop Categorical Variables
**The easiest approach** to dealing with categorical variables is to simply remove them from the dataset. This approach **will only work well if the columns did not contain useful information**.

~~~
# ## ScoreApproach 1 (Drop Categorical Variables)
drop_train_x = train_x.select_dtypes(exclude=['object'])
drop_vtion_x = vtion_x.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
>>> MAE from Approach 1 (Drop categorical variables):
>>> 175703.48185157913
~~~

### Label Encoding
Label encoding assigns each category to a unique integer.
![label encoding](label-encoding.png)
This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables. For tree-based models (like decision trees and random forests), you can expect **label encoding to work well with ordinal variables**.

~~~
# ## Score from Approach 2 (Label Encoding)
from sklearn.preprocessing import LabelEncoder

# make a copy to avoid changing original data
label_train_x = train_x.copy()
label_vtion_x = vtion_x.copy()

# get all the categorical data
s = (train_x.dtypes == 'object')
object_cols = list(s[s].index)

# apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
  label_train_x[col] = label_encoder.fit_transform(label_train_x[col])
  label_vtion_x[col] = label_encoder.transform(label_vtion_x[col])

print("MAE from Approach 2 (Label Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
>>> MAE from Approach 2 (Label Encoding):
>>> 165936.40548390493
~~~
In the code cell above, for each column, we randomly assign each unique value to a different integer. This is a common approach that is simpler than providing custom labels; however, we can expect an additional boost in performance if we provide better-informed labels for all ordinal variables.



### One-Hot Encoding
One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data.
![One-Hot encoding](one-hot_encoding.png)
In the original dataset, "Color" is a categorical variable with three categories: "Red", "Yellow", and "Green". The corresponding one-hot encoding contains one column for each possible value. If the value was "Yellow", we put a 1 in the "Yellow" column, if the value was "Red", we put a 1 in the "Red" column and so on.

In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data (e.g., "Red" is neither more nor less than "Yellow"). We refer to categorical variables without an intrinsic ranking as **nominal variables**. One-hot encoding generally does not perform well if the categorical variable takes on a large number of values (i.e., you generally **won't use it for variables taking more than 15 different values**).

~~~
from sklearn.preprocessing import OneHotEncoder

# ## Score from Approach 3 (One-Hot Encoding)

# get all the categorical data
s = (train_x.dtypes == 'object')
object_cols = list(s[s].index)

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_x[object_cols])
OH_cols_vtion = pd.DataFrame(OH_encoder.transform(vtion_x[object_cols])

# One-Hot encoding removed index; put it back
OH_cols_train.index = train_x.index
OH_cols_vtion.index = vtion_x.index

# Remove categorical columns (will replace with one-hot encoding)
num_train = train_x.drop(object_cols, axis=1)
num_vtion = vtion_x.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_train_x = pd.concat([num_train, OH_cols_train], axis=1)
OH_vtion_x = pd.concat([num_vtion, OH_cols_vtion], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
>>> MAE from Approach 3 (One-Hot Encoding):
>>> 166089.4893009678
~~~
We use the OneHotEncoder class from scikit-learn to get one-hot encodings. There are a number of parameters that can be used to customize its behavior.

We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
To use the encoder, we supply only the categorical columns that we want to be one-hot encoded. For instance, to encode the training data, we supply X_train[object_cols]. (object_cols in the code cell below is a list of the column names with categorical data, and so X_train[object_cols] contains all of the categorical data in the training set.)


### Example
Next, we obtain a list of all of the categorical variables in the training data.

We do this by checking the data type (or dtype) of each column. The object dtype indicates a column has text (there are other things it could theoretically be, but that's unimportant for our purposes). For this dataset, the columns with text indicate categorical variables.
~~~
s = (train_x.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
~~~




# Pipelines
A pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

Pipelines have some important benefits :
  * Cleaner Code: No need to manually keep track of your training and validation data at each step.
  * Fewer Bugs: There are fewer opportunities to misapply a step or forget a preprocessing step.
  * Easier to Productionize.
  * More Options for Model Validation.

We will work with the Melbourne Housing dataset. Imagine you already have the training and validation data in X_train, X_valid, y_train, and y_valid. **Construct pipeline in three steps**.


#### Step 1: Define Preprocessing Steps
Similar to how a pipeline bundles together preprocessing and modeling steps, we use the ColumnTransformer class to bundle together different preprocessing steps. The code below:

  + imputes missing values in numerical data, and
  + imputes missing values and applies a one-hot encoding to categorical data.

~~~
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
~~~

#### Step 2: Define the Model
Next, we define a random forest model with the familiar RandomForestRegressor class.

~~~
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
~~~

#### Step 3: Create and Evaluate the Pipeline
Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:

  + With the pipeline, we preprocess the training data and fit the model in a single line of code. (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)
  + With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions. (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)

~~~
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

>>> MAE: 160679.18917034855
~~~

#### Example
~~~
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


## code from the tutorial to preprocess the data and train a model


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

>>> MAE: 17861.780102739725

~~~



## Step 1: Improve the performance


~~~

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant') # Your code here

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
]) # Your code here

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0) # Your code here


# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

>>> MAE: 17621.3197260274


~~~

## Step 2: Generate test predictions

~~~


# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test) # Your code here


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

~~~





# Cross-Validation

Se enfrentará a decisiones sobre qué variables predictivas usar, qué tipos de modelos usar, qué argumentos proporcionar a esos modelos, etc. Hasta ahora, ha tomado estas decisiones de una manera basada en datos midiendo la calidad del modelo con una validación ( o reserva) establecido.

Pero hay algunos inconvenientes en este enfoque. Para ver esto, imagine que tiene un conjunto de datos con 5000 filas. Por lo general, conservará aproximadamente el 20% de los datos como un conjunto de datos de validación o 1000 filas. Pero esto deja algunas posibilidades al azar para determinar las puntuaciones del modelo. Es decir, un modelo podría funcionar bien en un conjunto de 1000 filas, incluso si sería inexacto en 1000 filas diferentes.

En un extremo, podría imaginar tener solo 1 fila de datos en el conjunto de validación. Si compara modelos alternativos, ¡cuál hace las mejores predicciones en un solo punto de datos será principalmente una cuestión de suerte!

En general, cuanto mayor sea el conjunto de validación, menor será la aleatoriedad (también conocida como "ruido") en nuestra medida de la calidad del modelo y más confiable será. Desafortunadamente, solo podemos obtener un gran conjunto de validación eliminando filas de nuestros datos de entrenamiento, y los conjuntos de datos de entrenamiento más pequeños significan peores modelos.


## What is cross-validation?¶
In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.

For example, we could begin by dividing the data into 5 pieces, each 20% of the full dataset. In this case, we say that we have broken the data into 5 "folds".

![cross-validation](cross-validation.png)

Then, we run one experiment for each fold:

  + In Experiment 1, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.
  + In Experiment 2, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
  + We repeat this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).

## When should you use cross-validation?
Cross-validation gives a more accurate measure of model quality, which is especially important if you are making a lot of modeling decisions. However, it can take longer to run, because it estimates multiple models (one for each fold).

So, given these tradeoffs, when should you use each approach?

  + For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
  + For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

There's no simple threshold for what constitutes a large vs. small dataset. But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation.

Alternatively, you can run cross-validation and see if the scores for each experiment seem close. If each experiment yields the same results, a single validation set is probably sufficient.


## Example
Then, we define a pipeline that uses an imputer to fill in missing values and a random forest model to make predictions.

While it's possible to **do cross-validation without pipelines, it is quite difficult!** Using a pipeline will make the code remarkably straightforward.

~~~
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[ ('preprocessor', SimpleImputer()) , ('model', RandomForestRegressor(n_estimators=50,  random_state=0)) ])
~~~

**We obtain the cross-validation scores with the cross_val_score() function from scikit-learn. We set the number of folds with the cv parameter.**

~~~
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = - 1 * cross_val_score( my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
>>> MAE scores:
 [301628.7893587  303164.4782723  287298.331666   236061.84754543 260383.45111427]
~~~


The scoring parameter chooses a measure of model quality to report: in this case, we chose negative mean absolute error (MAE). The docs for scikit-learn show a list of options.

It is a little surprising that we specify negative MAE. Scikit-learn has a convention where all metrics are defined so a high number is better. Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere.

We typically want a single measure of model quality to compare alternative models. So we take the average across experiments.


~~~
print("Average MAE score (across experiments):")
print(scores.mean())

>>> Average MAE score (across experiments):
>>> 277707.3795913405
~~~

#### Conclusion
Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code: note that we no longer need to keep track of separate training and validation sets. So, especially for small datasets, it's a good improvement!



## Exercise

~~~

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()




from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])




from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())
>>> Average MAE score: 18276.410356164386


~~~

### Step 1: Write a useful function
A function get_score() that reports the average (over three cross-validation folds) MAE of a machine learning pipeline that uses:
the data in X and y to create folds,
SimpleImputer() (with all parameters left as default) to replace missing values, and
RandomForestRegressor() (with random_state=0) to fit a random forest model.
The n_estimators parameter supplied to get_score() is used when setting the number of trees in the random forest model.

~~~
def get_score(n_estimators):
    my_pipeline = Pipeline( steps=[ ('preprocessor', SimpleImputer()) , ('model', RandomForestRegressor(n_estimators, random_state=0)) ])
    scores = - 1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()
~~~

### Step 2: Test different parameter values
Now, you will use the function that you defined in Step 1 to evaluate the model performance corresponding to eight different values for the number of trees in the random forest: 50, 100, 150, ..., 300, 350, 400.

Store your results in a Python dictionary results, where results[i] is the average MAE returned by get_score(i).

~~~
results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)
~~~

Use the next cell to visualize your results from Step 2. Run the code without changes.

~~~
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()
~~~

### Step 3: Find the best parameter value
Given the results, which value for n_estimators seems best for the random forest model? Use your answer to set the value of n_estimators_best.

~~~
n_estimators_best = n_estimators_best = min(results, key=results.get)
~~~


### Conclusion
In this exercise, you have explored one method for choosing appropriate parameters in a machine learning model.

If you'd like to learn more about hyperparameter optimization, you're encouraged to start with grid search, which is a straightforward method for determining the best combination of parameters for a machine learning model. Thankfully, scikit-learn also contains a built-in function GridSearchCV() that can make your grid search code very efficient!




# XGBoost
 build and optimize models with **Gradient Boosting**. This method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.

Random forest method achieves better performance than a single decision tree simply by averaging the predictions of many decision trees.

We refer to the random forest method as an "ensemble method". By definition, ensemble methods combine the predictions of several models (e.g., several trees, in the case of random forests)

Another ensemble method called gradient boosting.

## Gradient Boosting
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.

In this tutorial, you will learn how to build and optimize models with gradient boosting. This method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.

Introduction
For much of this course, you have made predictions with the random forest method, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees.

We refer to the random forest method as an "ensemble method". By definition, ensemble methods combine the predictions of several models (e.g., several trees, in the case of random forests).

Next, we'll learn about another ensemble method called gradient boosting.

Gradient Boosting
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.

It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

Then, we start the cycle:

  * First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
  * These predictions are used to calculate a loss function (like mean squared error, for instance).
  * Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
  * Finally, we add the new model to ensemble, and ...
... repeat!

![Gradient boosting](gradient-boosting.png)


## Example (with the XGBoost)

We begin by loading the training and validation data in X_train, X_valid, y_train, and y_valid.
~~~
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
~~~

In this example, you'll work with the XGBoost library. XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed. (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.)

In the next code cell, we import the scikit-learn API for XGBoost (xgboost.XGBRegressor). This allows us to build and fit a model just as we would in scikit-learn. As you'll see in the output, the XGBRegressor class has many tunable parameters -- you'll learn about those soon!
~~~
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
~~~

We also make predictions and evaluate the model.
~~~
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

>>> Mean Absolute Error: 239960.14714193667
~~~

### Parameter Tuning
XGBoost has a few parameters that can dramatically affect accuracy and training speed. The first parameters you should understand are:

##### n_estimators
n_estimators specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.

  + Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
  + Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).

Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

Here is the code to set the number of models in the ensemble:
~~~
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
~~~

##### early_stopping_rounds
early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.

We can modify the example above to include early stopping:
~~~
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
~~~
If you later want to fit a model with all of your data, set n_estimators to whatever value you found to be optimal when run with early stopping.

##### learning_rate
Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.

This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.

In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets learning_rate=0.1.

Modifying the example above to change the learning rate yields the following code:
~~~
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
~~~

##### n_jobs
On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't help.

The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.

Here's the modified example:
~~~
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
~~~

## Conclusion
**XGBoost is a the leading software library for working with standard tabular data (the type of data you store in Pandas DataFrames**, as opposed to more exotic types of data like images and videos). With careful parameter tuning, you can train highly accurate models.


## Exercise: XGBoost

### Load the data
~~~
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
~~~

### Step 1: Build model (build & train gradient boosting model)
  * Part A
    + set my_model_1 to an XGBoost model, set random seed to 0 (random_state=0). other parameters as default.
    + fit the model with training data from X_train and y_train.

~~~
# Define the model
my_model_1 = XGBRegressor(random_state=0)

# Fit the model
my_model_1.fit(X_train, y_train)
~~~

  * Part B
    + Set predictions_1 to the model's predictions for the validation data. Recall that the validation features are stored in X_valid.

~~~
predictions_1 = my_model_1.predict(X_valid)
~~~

  * Part C
    + Finally, use the mean_absolute_error() function to calculate the mean absolute error (MAE) corresponding to the predictions for the validation set. Recall that the labels for the validation data are stored in y_valid.

~~~
# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid)
print("Mean Absolute Error:" , mae_1)
~~~

### Step 2: Improve the model
Now that you've trained a default model as baseline, it's time to tinker with the parameters, to see if you can get better performance!

  + Begin by setting my_model_2 to an XGBoost model, using the XGBRegressor class.
  + Then, fit the model to the training data in X_train and y_train.
  + Set predictions_2 to the model's predictions for the validation data. Recall that the validation features are stored in X_valid.
  + Finally, use the mean_absolute_error() function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set. Recall that the labels for the validation data are stored in y_valid.

In order for this step to be marked correct, your model in my_model_2 must attain lower MAE than the model in my_model_1.
~~~
# Define the model
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# Fit the model
my_model_2.fit(X_train, y_train)

# Get predictions
predictions_2 = my_model_2.predict(X_valid)

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:" , mae_2)
~~~

### Step 3: Break the model
In this step, you will create a model that performs worse than the original model in Step 1. This will help you to develop your intuition for how to set parameters. You might even find that you accidentally get better performance, which is ultimately a nice problem to have and a valuable learning experience!

  + Begin by setting my_model_3 to an XGBoost model, using the XGBRegressor class.
  + Then, fit the model to the training data in X_train and y_train.
  + Set predictions_3 to the model's predictions for the validation data. Recall that the validation features are stored in X_valid.
  + Finally, use the mean_absolute_error() function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set. Recall that the labels for the validation data are stored in y_valid.

In order for this step to be marked correct, your model in my_model_3 must attain higher MAE than the model in my_model_1.
~~~
# Define the model
my_model_3 = XGBRegressor(n_estimators=1)

# Fit the model
my_model_3.fit(X_train, y_train)

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)
print("Mean Absolute Error:" , mae_3)
~~~



# Data Leakage
Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.

**Leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.**

**There are two main types of leakage: target leakage and train-test contamination**.

## Target leakage
Target leakage occurs when your predictors include data that will not be available at the time you make predictions. It is important to think about target leakage in terms of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions.

Imagine you want to predict who will get sick with pneumonia. People take antibiotic medicines after getting pneumonia (to recover). took_antibiotic_medicine is frequently changed after the value for got_pneumonia is determined. This is target leakage.

The model would see that anyone who has a value of False for took_antibiotic_medicine didn't have pneumonia. Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.

But the model will be very inaccurate when subsequently deployed in the real world, because even **patients who will get pneumonia won't have received antibiotics yet** when we need to make predictions about their future health.

**To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded**.

![target-leakage](target-leakage.png)


## Train-Test Contamination
A different type of leak occurs when you aren't careful to distinguish training data from validation data.

Recall that validation is meant to be a measure of how the model does on data that it hasn't considered before. You can corrupt this process in subtle ways if the validation data affects the preprocessing behavior. This is sometimes called train-test contamination.

**For example, imagine you run preprocessing (like fitting an imputer for missing values) before calling train_test_split(). The end result? Your model may get good validation scores, giving you great confidence in it, but perform poorly when you deploy it to make decisions**.

After all, you incorporated data from the validation or test data into how you make predictions, so may do well on that particular data even if it can't generalize to new data. This problem becomes even more subtle (and more dangerous) when you do more complex feature engineering.

If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, including the fitting of preprocessing steps. This is easier if you use scikit-learn pipelines. When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline!


## Example
One way to detect and remove target leakage.

We will use a dataset about credit card applications. The end result is that information about each credit card application is stored in a DataFrame X. We'll use it to predict which applications were accepted in a Series y.
~~~
import pandas as pd

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv',
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()
~~~

Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.
~~~
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y,
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())

>>> Cross-validation accuracy: 0.978776
~~~

**With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for target leakage**.

Here is a summary of the data, which you can also find under the data tab:

  + card: 1 if credit card application accepted, 0 if not
  + reports: Number of major derogatory reports
  + age: Age n years plus twelfths of a year
  + income: Yearly income (divided by 10,000)
  + share: Ratio of monthly credit card expenditure to yearly income
  + expenditure: Average monthly credit card expenditure
  + owner: 1 if owns home, 0 if rents
  + selfempl: 1 if self-employed, 0 if not
  + dependents: 1 + number of dependents
  + months: Months living at current address
  + majorcards: Number of major credit cards held
  + active: Number of active credit accounts

A few variables look suspicious. For example, does expenditure mean expenditure on this card or on cards used before appying?

At this point, basic data comparisons can be very helpful:
~~~
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \ %(( expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \ %(( expenditures_cardholders == 0).mean()))

>>> Fraction of those who did not receive a card and had no expenditures: 1.00
>>> Fraction of those who received a card and had no expenditures: 0.02
~~~

As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. It's not surprising that our model appeared to have a high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

Since share is partially determined by expenditure, it should be excluded too. The variables active and majorcards are a little less clear, but from the description, they sound concerning. In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.

We would run a model without target leakage as follows:
~~~
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y,
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())

>>> Cross-val accuracy: 0.828650
~~~
This accuracy is quite a bit lower, which might be disappointing. However, we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model would likely do much worse than that (in spite of its higher apparent score in cross-validation).


## Conclusion
Data leakage can be multi-million dollar mistake in many data science applications. **Careful separation of training and validation data can prevent train-test contamination, and pipelines can help implement this separation**. Likewise, a combination of caution, common sense, and data exploration can help identify target leakage.



## Exercise: Data Leakage

### Step 1: The Data Science of Shoelaces
Nike has hired you as a data science consultant to help them save money on shoe materials. Your first assignment is to review a model one of their employees built to predict how many shoelaces they'll need each month. The features going into the machine learning model include:

  * The current month (January, February, etc)
  * Advertising expenditures in the previous month
  * Various macroeconomic features (like the unemployment rate) as of the beginning of the current month
  * The amount of leather they ended up using in the current month

The results show the model is almost perfectly accurate if you include the feature about how much leather they used. But it is only moderately accurate if you leave that feature out. You realize this is because the amount of leather they use is a perfect indicator of how many shoes they produce, which in turn tells you how many shoelaces they need.

Do you think the leather used feature constitutes a source of data leakage? If your answer is "it depends," what does it depend on?

This is tricky, and it **depends on details of how data is collected (which is common when thinking about leakage)**. *Would you at the beginning of the month decide how much leather will be used that month? If so, this is ok. But if that is determined during the month, you would not have access to it when you make the prediction.* If you have a guess at the beginning of the month, and it is subsequently changed during the month, the actual amount used during the month cannot be used as a feature (because it causes leakage).


### Step 2: Return of the Shoelaces
You have a new idea. You could use the amount of leather Nike ordered (rather than the amount they actually used) leading up to a given month as a predictor in your shoelace model.

Does this change your answer about whether there is a leakage problem? If you answer "it depends," what does it depend on?

This could be fine, but it depends on whether they *order shoelaces first or leather first*. If they order shoelaces first, you won't know how much leather they've ordered when you predict their shoelace needs. If they order leather first, then you'll have that number available when you place your shoelace order, and you should be ok.


### Step 3: Getting Rich With Cryptocurrencies?
Your friend, who is also a data scientist, says he has built a model that will let you turn your money into millions of dollars. Specifically, his model predicts the price of a new cryptocurrency (like Bitcoin, but a newer one) one day ahead of the moment of prediction. His plan is to purchase the cryptocurrency whenever the model says the price of the currency (in dollars) is about to go up.

The most important features in his model are:
  + Current price of the currency
  + Amount of the currency sold in the last 24 hours
  + Change in the currency price in the last 24 hours
  + Change in the currency price in the last 1 hour
  + Number of new tweets in the last 24 hours that mention the currency

The value of the cryptocurrency in dollars has fluctuated up and down by over  100𝑖𝑛𝑡ℎ𝑒𝑙𝑎𝑠𝑡𝑦𝑒𝑎𝑟,𝑎𝑛𝑑𝑦𝑒𝑡ℎ𝑖𝑠𝑚𝑜𝑑𝑒𝑙′𝑠𝑎𝑣𝑒𝑟𝑎𝑔𝑒𝑒𝑟𝑟𝑜𝑟𝑖𝑠𝑙𝑒𝑠𝑠𝑡ℎ𝑎𝑛 1. He says this is proof his model is accurate, and you should invest with him, buying the currency whenever the model says it is about to go up.

Is he right? If there is a problem with his model, what is it?

There is no source of leakage here. These features should be available at the moment you want to make a predition, and they're unlikely to be changed in the training data after the prediction target is determined. But, the way he describes accuracy could be misleading if you aren't careful. If the price moves gradually, today's price will be an accurate predictor of tomorrow's price, but it may not tell you whether it's a good time to invest. For instance, if it is  100𝑡𝑜𝑑𝑎𝑦,𝑎𝑚𝑜𝑑𝑒𝑙𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑛𝑔𝑎𝑝𝑟𝑖𝑐𝑒𝑜𝑓 100 tomorrow may seem accurate, even if it can't tell you whether the price is going up or down from the current price. A better prediction target would be the change in price over the next day. If you can consistently predict whether the price is about to go up or down (and by how much), you may have a winning investment opportunity.


### Step 4: Preventing Infections
An agency that provides healthcare wants to predict which patients from a rare surgery are at risk of infection, so it can alert the nurses to be especially careful when following up with those patients.

You want to build a model. Each row in the modeling dataset will be a single patient who received the surgery, and the prediction target will be whether they got an infection.

Some surgeons may do the procedure in a manner that raises or lowers the risk of infection. But how can you best incorporate the surgeon information into the model?

You have a clever idea.

  1. Take all surgeries by each surgeon and calculate the infection rate among those surgeons.
  2. For each patient in the data, find out who the surgeon was and plug in that surgeon's average infection rate as a feature.

Does this pose any target leakage issues? Does it pose any train-test contamination issues?


This poses a risk of both target leakage and train-test contamination (though you may be able to avoid both if you are careful).

You have target leakage if a given patient's outcome contributes to the infection rate for his surgeon, which is then plugged back into the prediction model for whether that patient becomes infected. You can avoid target leakage if you calculate the surgeon's infection rate by using only the surgeries before the patient we are predicting for. Calculating this for each surgery in your training data may be a little tricky.

You also have a train-test contamination problem if you calculate this using all surgeries a surgeon performed, including those from the test-set. The result would be that your model could look very accurate on the test set, even if it wouldn't generalize well to new patients after the model is deployed. This would happen because the surgeon-risk feature accounts for data in the test set. Test sets exist to estimate how the model will do when seeing new data. So this contamination defeats the purpose of the test set.


### Step 5: Housing Prices
You will build a model to predict housing prices. The model will be deployed on an ongoing basis, to predict the price of a new house when a description is added to a website. Here are four features that could be used as predictors.

  1. Size of the house (in square meters)
  2. Average sales price of homes in the same neighborhood
  3. Latitude and longitude of the house
  4. Whether the house has a basement

You have historic data to train and validate the model.

Which of the features is most likely to be a source of leakage?


2 is the source of target leakage. Here is an analysis for each feature:

  1. The size of a house is unlikely to be changed after it is sold (though technically it's possible). But typically this will be available when we need to make a prediction, and the data won't be modified after the home is sold. So it is pretty safe.

  2. We don't know the rules for when this is updated. If the field is updated in the raw data after a home was sold, and the home's sale is used to calculate the average, this constitutes a case of target leakage. At an extreme, if only one home is sold in the neighborhood, and it is the home we are trying to predict, then the average will be exactly equal to the value we are trying to predict. In general, for neighborhoods with few sales, the model will perform very well on the training data. But when you apply the model, the home you are predicting won't have been sold yet, so this feature won't work the same as it did in the training data.

  3. These don't change, and will be available at the time we want to make a prediction. So there's no risk of target leakage here.

  4. This also doesn't change, and it is available at the time we want to make a prediction. So there's no risk of target leakage here.



## Conclusion
Leakage is a hard and subtle issue. You should be proud if you picked up on the issues in these examples.









#
