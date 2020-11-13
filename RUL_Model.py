from keras.models import load_model
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Scale Test
def scale(df):
    #return (df - df.mean())/df.std()
    return (df - df.min())/(df.max()-df.min())

# Load Model
model = load_model('ClassifierV2_Smoothing01.h5')
w = [200,175,150,125,100,75,50,40,30,25,20,15,10,5] #Bin definitions associated with current models
        
##Create dictionary of category labels to bin midpoints:
rul_cats = [210]
for i in range(len(w)-1):
    upper , lower = w[i], w[i+1]
    rul_cats.append(round(np.mean([upper,lower])))
rul_cats.append(3)
rul_dict = dict(zip(range(len(w)+1), rul_cats))

### Generate Sequences ### 
sequence_length = 50

def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 (start stop) -> from row 0 to row 50
    # 1 51 (start stop) -> from row 1 to row 51
    # 2 52 (start stop) -> from row 2 to row 52
    # ...
    # 141 191 (start stop) -> from row 141 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]



def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def pad_series(test_df, sequence_length=sequence_length):
	groupby = test_df.groupby('id')['cycle'].max()
	nrows = groupby[groupby<=50]
	over_50 = test_df[~test_df.id.isin(nrows.index)]

	for unit in nrows.index:
	    temp = test_df[test_df.id == unit]
	    padding = pd.DataFrame()
	    nmissing = 51 - len(temp)
	    #Create synthetic starter rows
	    for i in range(nmissing):
	        padding = padding.append(pd.DataFrame(temp.iloc[0]).transpose(), ignore_index=True)
	    #Combine synthetic starter padding with available data
	    temp = padding.append(temp, ignore_index=True)
	    #Renumber cycles
	    temp.cycle = range(1,len(temp)+1)
	    #Append new padded series to over 50, 
	    over_50 = over_50.append(temp, ignore_index=True)
	#Reorder dataframe by id and cycle
	over_50 = over_50.sort_values(by=['id', 'cycle'])
	return over_50

def preprocess(test_df):
	### Rename and Parse raw columns
	if len(test_df.columns == 28):
		test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
	test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
	                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
	                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
	### SCALE TEST DATA ###
	for col in test_df.columns:
	    if col[0] == 's':
	        test_df[col] = scale(test_df[col])
	test_df = test_df.dropna(axis=1)

	### Pad Sequences with under 51 cycles
	test_df = pad_series(test_df)

	### GENERATE X TEST ###
	x_test = []
	sequence_cols = ['setting1', 'setting2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
	for engine_id in test_df.id.unique():

	    for sequence in gen_sequence(test_df[test_df.id==engine_id], sequence_length, sequence_cols):
	        x_test.append(sequence)
	    
	x_test = np.asarray(x_test)

	### TRANSFORM X TRAIN TEST IN IMAGES ###

	x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')

	return test_df, x_test_img

# Generate Class Predictions
def predict(x_test_img):
	"""Returns the class labels, one for each sliding window."""
	return model.predict_classes(x_test_img)

def preprocess_and_predict(test_df, output='RUL', sequence_length=sequence_length):
	"""Returns one prediction per unit. Output can be 'RUL' or 'Category'."""
	test_df, x_test_img = preprocess(test_df)
	class_predictions = predict(x_test_img)
	test_df = test_df[test_df['cycle']>sequence_length]
	test_df['class_prediction'] = class_predictions
	test_df['RUL_prediction'] = test_df['class_prediction'].map(rul_dict)
	return test_df

def summarize_predictions_by_unit(test_df_output, col='RUL_prediction'):
	"""Input should already have class_prediction and RUL_prediction columns.
	You can chain this with preprocess_and_predict. For example:
	rul_predictions = summarize_predictions_by_unit(preprocess_and_predict(test_df))
	"""
	return test_df_output.groupby('id').tail(1)[col].values
