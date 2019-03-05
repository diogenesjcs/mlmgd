import shutil
import os
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import itertools
tf.logging.set_verbosity(tf.logging.INFO)

mydir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
mydir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model2")

if os.path.exists(mydir) and os.path.isdir(mydir):
    shutil.rmtree(mydir)
if os.path.exists(mydir2) and os.path.isdir(mydir2):
    shutil.rmtree(mydir2)


config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
# (nothing gets printed in Jupyter, only if you run it standalone)

data = pd.read_csv("data/data.csv")
dataPredict = pd.read_csv("data/predict.csv")


data['yr_release'] = pd.to_numeric(
    data.releasedate.str.slice(0, 4))
data['month_release'] = pd.to_numeric(
    data.releasedate.str.slice(5, 7))
data['day_release'] = pd.to_numeric(
    data.releasedate.str.slice(8, 10))
data['beforedate'] = pd.to_datetime(data['beforedate'])
data['date'] = pd.to_datetime(data['date'])
data['nextdate'] = pd.to_datetime(data['nextdate'])


data['beforedays'] = (data['date'] - data['beforedate']).dt.days
data['nextdays'] = (data['nextdate'] - data['date']).dt.days
data['releasedays'] = (data['date'] - pd.to_datetime(data.releasedate)).dt.days
data['releasedays'] = np.array(data['releasedays']).clip(0)
data['beforedays'] = np.log((1 + data['beforedays']))

data['value'] = np.log((1 + data['value']))
data['beforevalue'] = np.log((1 + data['beforevalue']))
data['maxvalue'] = np.log((1 + data['maxvalue']))
data['minvalue'] = np.log((1 + data['minvalue']))
data['releasedays'] = np.log((1 + data['releasedays']))
data['rating'] = np.log((1 + data['rating']))
data['id'] = np.log((1 + data['id']))

dataPredict['yr_release'] = pd.to_numeric(
    dataPredict.releasedate.str.slice(0, 4))
dataPredict['month_release'] = pd.to_numeric(
    dataPredict.releasedate.str.slice(5, 7))
dataPredict['day_release'] = pd.to_numeric(
    dataPredict.releasedate.str.slice(8, 10))
dataPredict['beforedate'] = pd.to_datetime(dataPredict['beforedate'])
dataPredict['date'] = pd.to_datetime(dataPredict['date'])
dataPredict['nextdate'] = pd.to_datetime(dataPredict['nextdate'])


dataPredict['beforedays'] = (
    dataPredict['date'] - dataPredict['beforedate']).dt.days
dataPredict['nextdays'] = (dataPredict['nextdate'] -
                           dataPredict['date']).dt.days
dataPredict['releasedays'] = (
    dataPredict['date'] - pd.to_datetime(dataPredict.releasedate)).dt.days
dataPredict['releasedays'] = np.array(dataPredict['releasedays']).clip(0)
dataPredict['beforedays'] = np.log((1 + dataPredict['beforedays']))

dataPredict['value'] = np.log((1 + dataPredict['value']))
dataPredict['beforevalue'] = np.log((1 + dataPredict['beforevalue']))
dataPredict['maxvalue'] = np.log((1 + dataPredict['maxvalue']))
dataPredict['minvalue'] = np.log((1 + dataPredict['minvalue']))
dataPredict['releasedays'] = np.log((1 + dataPredict['releasedays']))
dataPredict['rating'] = np.log((1 + dataPredict['rating']))
dataPredict['id'] = np.log((1 + dataPredict['id']))


features = pd.DataFrame(data, columns=[
    'id', 'releasedays', 'rating', 'storeId', 'minvalue', 'maxvalue', 'beforevalue', 'beforedays', 'value'])

featuresPredict = pd.DataFrame(dataPredict, columns=[
    'id', 'releasedays', 'rating', 'storeId', 'minvalue', 'maxvalue', 'beforevalue', 'beforedays', 'value'])


featuresPredict1 = pd.DataFrame(dataPredict, columns=['nextvalue'])
featuresPredict2 = pd.DataFrame(dataPredict, columns=['nextdays'])

scal4 = RobustScaler()
scal4.fit(featuresPredict)
featuresPredict = pd.DataFrame(data=scal4.transform(
    featuresPredict), columns=featuresPredict.columns, index=featuresPredict.index)

featuresPredict['storeId'] = pd.DataFrame(data=scal4.inverse_transform(
    featuresPredict), columns=featuresPredict.columns, index=featuresPredict.index)['storeId'].astype(np.int64)


labels = pd.DataFrame(data, columns=['nextvalue'])
labels2 = pd.DataFrame(data, columns=['nextdays'])


X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0, random_state=101)

scaler = RobustScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(
    X_train), columns=X_train.columns, index=X_train.index)
X_train['storeId'] = pd.DataFrame(data=scaler.inverse_transform(
    X_train), columns=X_train.columns, index=X_train.index)['storeId'].astype(np.int64)

""" scal = RobustScaler()
scal.fit(X_test)
X_test = pd.DataFrame(data=scal.transform(
    X_test), columns=X_test.columns, index=X_test.index)

X_test['storeId'] = pd.DataFrame(data=scal.inverse_transform(
    X_test), columns=X_test.columns, index=X_test.index)['storeId'].astype(np.int64)

scal2 = RobustScaler()
scal2.fit(y_test)

y_test = pd.DataFrame(data=scal2.transform(
    y_test), columns=y_test.columns, index=y_test.index) """

y_train = pd.DataFrame(
    data=y_train, columns=y_train.columns, index=X_train.index)

type(X_train), type(y_train)
""" 
type(X_test), type(y_test)
 """

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    features, labels2, test_size=0, random_state=101)

scaler21 = RobustScaler()
scaler21.fit(X_train2)
X_train2 = pd.DataFrame(data=scaler21.transform(
    X_train2), columns=X_train2.columns, index=X_train2.index)
X_train2['storeId'] = pd.DataFrame(data=scaler21.inverse_transform(
    X_train2), columns=X_train2.columns, index=X_train2.index)['storeId'].astype(np.int64)

y_train2 = pd.DataFrame(
    data=y_train2, columns=y_train2.columns, index=X_train2.index)

type(X_train2), type(y_train2)
""" 
type(X_test2), type(y_test2) """

type(featuresPredict)

rating = tf.feature_column.numeric_column(
    'rating')
storeId = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_identity('storeId', num_buckets=6))
minvalue = tf.feature_column.numeric_column(
    'minvalue')
maxvalue = tf.feature_column.numeric_column(
    'maxvalue')
beforevalue = tf.feature_column.numeric_column(
    'beforevalue')
beforedays = tf.feature_column.numeric_column(
    'beforedays')
value = tf.feature_column.numeric_column(
    'value')
releasedays = tf.feature_column.numeric_column(
    'releasedays')
id = tf.feature_column.numeric_column(
    'id')

feat_cols = {id: id, releasedays: features['releasedays'],
             rating: features['rating'], storeId: features['storeId'], minvalue: features['minvalue'], maxvalue: features['maxvalue'], beforevalue: features['beforevalue'], beforedays: features['beforedays'], value: features['value']}


input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train['nextvalue'], batch_size=10000, num_epochs=300, shuffle=True)

input_func2 = tf.estimator.inputs.pandas_input_fn(
    x=X_train2, y=y_train2['nextdays'], batch_size=10000, num_epochs=300, shuffle=True)

""" 
eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, y=y_test['nextvalue'], batch_size=10000, num_epochs=1, shuffle=False) """


dnn_model = tf.estimator.DNNRegressor(label_dimension=1, model_dir=mydir,
                                      hidden_units=[30, 30], feature_columns=feat_cols, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss_reduction=tf.losses.Reduction.MEAN)

dnn_model2 = tf.estimator.DNNRegressor(label_dimension=1, model_dir=mydir2,
                                       hidden_units=[30, 30], feature_columns=feat_cols, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss_reduction=tf.losses.Reduction.MEAN)

dnn_model.train(input_fn=input_func)
dnn_model2.train(input_fn=input_func2)


"""
evaluation = dnn_model.evaluate(input_fn=eval_input_func)

print(evaluation['loss'])

predictions = dnn_model.predict(input_fn=eval_input_func)
pred = list(predictions)

predicted_vals = []

for pred in dnn_model.predict(input_fn=eval_input_func):
    predicted_vals.append(pred['predictions'])


mse = mean_squared_error(predicted_vals, y_test)
print('Mean Squared Error [DNNRegrssor]: ', mse)


reality = np.expm1(np.array(scal2.inverse_transform(y_test)))
print(reality)


predictions = np.expm1(np.array(scal2.inverse_transform(predicted_vals)))
print(predictions) """


eval_input_func_predict = tf.estimator.inputs.pandas_input_fn(
    x=featuresPredict, y=featuresPredict1['nextvalue'], batch_size=10000, num_epochs=1, shuffle=False)

eval_input_func_predict2 = tf.estimator.inputs.pandas_input_fn(
    x=featuresPredict, y=featuresPredict2['nextdays'], batch_size=10000, num_epochs=1, shuffle=False)

predictions = dnn_model.predict(input_fn=eval_input_func_predict)
pred = list(predictions)

predicted_vals = []

for pred in dnn_model.predict(input_fn=eval_input_func_predict):
    predicted_vals.append(pred['predictions'])

predictions = predicted_vals

dataPredict['nextvalue'] = predicted_vals

predictions2 = dnn_model2.predict(input_fn=eval_input_func_predict2)
pred2 = list(predictions2)

predicted_vals2 = []

for pred2 in dnn_model2.predict(input_fn=eval_input_func_predict2):
    predicted_vals2.append(pred2['predictions'])

predictions2 = predicted_vals2

dataPredict['nextdays'] = predicted_vals2


resultado = pd.DataFrame(dataPredict, columns=[
    'gameId', 'storeId', 'nextvalue', 'nextdays'])


resultado.to_csv(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data", "resultado.csv"), sep=',', encoding='utf-8')
