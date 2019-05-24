import os
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, concatenate, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()

data_folder = args.data_folder
print('training dataset is stored here:', data_folder)

raw = pd.read_csv(data_folder + '/train.csv')

print(raw.shape)

# Data Engineering
df = raw

#Impute missing values
df.Product_Category_2.fillna(-1, inplace=True)
df.Product_Category_3.fillna(-1, inplace=True)

df.User_ID = df.User_ID.astype(str)
df.Occupation = df.Occupation.astype(str)
df.Marital_Status = df.Marital_Status.astype(str)
df.Product_Category_1 = df.Product_Category_1.astype(str)
df.Product_Category_2 = df.Product_Category_2.astype(str)
df.Product_Category_3 = df.Product_Category_3.astype(str)

target_col = 'Purchase'
cat_cols = [col for col in df.columns if col != target_col]

unique_col_count = df[cat_cols].nunique()

def cat2idx(dataset, cat_cols):
    data = dataset.copy()
    cat2idx_dict = dict()

    for col in cat_cols:
        unique_cat = data[col].unique()
        cat2idx_map = {o: i for i, o in enumerate(unique_cat)}
        data[col] = data[col].apply(lambda x: cat2idx_map[x])
        cat2idx_dict[col] = cat2idx_map

    return data, cat2idx_dict

df_indexed, index_map = cat2idx(df, cat_cols)

X_train, X_test, y_train, y_test = train_test_split(df_indexed[cat_cols].values, df_indexed[target_col].values,
                                                    test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

input_models=[]
output_embeddings=[]

for cat_col in cat_cols:
    cat_emb_name = cat_col + "_Embedding"
    cat_input_name = cat_col + "_Input"
    emb_size = int(min(np.ceil((unique_col_count[cat_col]) / 2), 50))

    input_model = Input(shape=(1,), name=cat_input_name)
    output_model = Embedding(unique_col_count[cat_col], emb_size, input_length=1, name=cat_emb_name)(input_model)
    output_model = Reshape(target_shape=(emb_size,))(output_model)

    input_models.append(input_model)
    output_embeddings.append(output_model)

x = concatenate(output_embeddings)
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.7)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.4)(x)
x = Dense(1)(x)

model = Model(inputs=input_models, outputs=x)
opt = Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])


X_train_list = []
X_test_list = []

for i, _ in enumerate(cat_cols):
    X_train_list.append(X_train[:, i])
    X_test_list.append(X_test[:, i])

run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])

model.fit(X_train_list, y_train, validation_data=(X_test_list, y_test), epochs=2 , batch_size=256)

pred = model.predict(X_test_list)
score = np.sqrt(mean_squared_error(y_test,pred))
run.log("Final RMSE", score)

os.makedirs('./outputs/model', exist_ok=True)
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")

with open('./outputs/model/index_map.pkl', 'wb') as f:
    pickle.dump(index_map, f)

with open('./outputs/model/outcome_var.pkl', 'wb') as f:
    pickle.dump(target_col, f)
