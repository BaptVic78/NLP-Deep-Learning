"""The six-emotion BiLSTM notebook, with model and evaluation saving."""
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from text_cleaning import normalize_text

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.keras.utils.set_random_seed(42)
root = Path(__file__).resolve().parent
out = root/'artifacts'; out.mkdir(exist_ok=True)
folder = root/'data'; folder.mkdir(exist_ok=True)
labels = ['sadness','joy','love','anger','fear','surprise']

# 1. The same dataset and official splits as the notebook.
for split in ['train','validation','test']:
    path = folder/f'emotion-{split}.parquet'
    if not path.exists():
        response = requests.get(f'https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split/{split}-00000-of-00001.parquet',timeout=90)
        response.raise_for_status(); path.write_bytes(response.content)
train = pd.read_parquet(folder/'emotion-train.parquet')
validation = pd.read_parquet(folder/'emotion-validation.parquet')
test = pd.read_parquet(folder/'emotion-test.parquet')
for df in [train,validation,test]:
    df['clean_text'] = df.text.map(lambda text:normalize_text(text,remove_numbers=True))
    conflicts = df.groupby('clean_text').label.nunique()
    df.drop(df[df.clean_text.isin(conflicts[conflicts>1].index)].index,inplace=True)
    df.drop_duplicates('clean_text',inplace=True)
    df.drop(df[df.clean_text==''].index,inplace=True)
validation = validation[~validation.clean_text.isin(train.clean_text)]
test = test[~test.clean_text.isin(train.clean_text)]
test = test[~test.clean_text.isin(validation.clean_text)]
samples = int(os.getenv('SAMPLES','0'))
if samples and samples<len(train):
    train,_ = train_test_split(train,train_size=samples,stratify=train.label,random_state=42)

# 2. Train-only vocabulary, then 256-token sequences.
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train.clean_text)
X_train = pad_sequences(tokenizer.texts_to_sequences(train.clean_text),maxlen=256,padding='post')
X_validation = pad_sequences(tokenizer.texts_to_sequences(validation.clean_text),maxlen=256,padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test.clean_text),maxlen=256,padding='post')

# 3. The notebook's two bidirectional LSTM layers and six-class softmax.
model = tf.keras.Sequential([
    Input(shape=(256,)), Embedding(10000,128),
    Bidirectional(LSTM(64,return_sequences=True)), Bidirectional(LSTM(32)),
    Dropout(.5), Dense(6,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# 4. Train with the official validation split.
options = tf.data.Options(); options.threading.private_threadpool_size=2
train_data = tf.data.Dataset.from_tensor_slices((X_train,train.label.to_numpy())).shuffle(len(train),seed=42).batch(32).with_options(options)
val_data = tf.data.Dataset.from_tensor_slices((X_validation,validation.label.to_numpy())).batch(32).with_options(options)
history = model.fit(train_data,validation_data=val_data,epochs=int(os.getenv('EPOCHS','10')),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)],shuffle=False,verbose=2)

# 5. Evaluate and save, without using test to select the model.
probabilities = []
for start in range(0,len(X_test),64):
    probabilities.extend(model(X_test[start:start+64],training=False).numpy())
predictions = np.argmax(np.array(probabilities),axis=1)
report = {'accuracy':float(accuracy_score(test.label,predictions)),
          'f1_macro':float(f1_score(test.label,predictions,average='macro')),
          'confusion':confusion_matrix(test.label,predictions,labels=list(range(6))).tolist(),
          'train_size':len(train),'validation_size':len(validation),'test_size':len(test),
          'sample_limit':samples,'epochs':len(history.history['loss']),'history':history.history,'labels':labels}
model.save(out/'emotions.keras')
(out/'emotions_tokenizer.json').write_text(tokenizer.to_json())
(out/'emotions_report.json').write_text(json.dumps(report,indent=2))
test[['text','label']].head(10).to_csv(out/'emotions_examples.csv',index=False)
print('Emotion test accuracy:',report['accuracy'],flush=True)
