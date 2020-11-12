from keras.models import load_model
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

# Scale Test
def scale(df):
    #return (df - df.mean())/df.std()
    return (df - df.min())/(df.max()-df.min())

# Load Model
model = load_model('ClassifierV2_Smoothing5.h5')
        

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


def preprocess(test_df):
	### Rename and Parse raw columns
	test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
	test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
	                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
	                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
	### SCALE TEST DATA ###

	for col in test_df.columns:
	    if col[0] == 's':
	        test_df[col] = scale(test_df[col])
	test_df = test_df.dropna(axis=1)

	### GENERATE X TEST ###
	x_test = []
	sequence_cols = ['setting1', 'setting2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
	for engine_id in test_df.id.unique():

	    for sequence in gen_sequence(test_df[test_df.id==engine_id], sequence_length, sequence_cols):
	        x_test.append(sequence)
	    
	x_test = np.asarray(x_test)

	### TRANSFORM X TRAIN TEST IN IMAGES ###

	x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')

	return x_test_img

# Generate Predictions
def predict(x_test_img):
	return model.predict_classes(x_test_img)

def preprocess_and_predict(test_df):
	x_test_img = preprocess(test_df)
	predictions = predict(x_test_img)
	return predictions
