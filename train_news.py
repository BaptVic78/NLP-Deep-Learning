"""Load → clean → tokenize → train the notebook BiLSTM → evaluate → save."""
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
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

# 1. Load and label the original two CSVs.
fake = pd.read_csv(root/'TP3-2-Fake-News-Detection/data/Fake.csv')
real = pd.read_csv(root/'TP3-2-Fake-News-Detection/data/True.csv')
fake['label'] = 0; real['label'] = 1
data = pd.concat([fake, real], ignore_index=True).dropna(subset=['title','text'])
data['clean_text'] = (data['title']+' '+data['text']).map(normalize_text)
conflicts = data.groupby('clean_text')['label'].nunique()
data = data[~data.clean_text.isin(conflicts[conflicts>1].index)].drop_duplicates('clean_text')
data = data[data.clean_text.str.len()>0]
samples = int(os.getenv('SAMPLES', '0'))
if samples and samples<len(data):
    data, _ = train_test_split(data, train_size=samples, stratify=data.label, random_state=42)
train, test = train_test_split(data, test_size=.2, stratify=data.label, random_state=42)
train, validation = train_test_split(train, test_size=.1, stratify=train.label, random_state=42)

# 2. Fit the vocabulary on train, then pad to the notebook's 256 tokens.
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train.clean_text)
X_train = pad_sequences(tokenizer.texts_to_sequences(train.clean_text), maxlen=256, padding='post')
X_validation = pad_sequences(tokenizer.texts_to_sequences(validation.clean_text), maxlen=256, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test.clean_text), maxlen=256, padding='post')

# 3. Keep the notebook architecture and its linear output (a logit).
model = tf.keras.Sequential([
    Input(shape=(256,)), Embedding(10000,128),
    Bidirectional(LSTM(64, return_sequences=True)), Bidirectional(LSTM(16)),
    Dense(64, activation='relu'), Dropout(.5), Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.0)])

# 4. Train, with validation and the notebook's early stopping.
options = tf.data.Options(); options.threading.private_threadpool_size = 2
train_data = tf.data.Dataset.from_tensor_slices((X_train, train.label.to_numpy())).shuffle(len(train), seed=42).batch(30).with_options(options)
val_data = tf.data.Dataset.from_tensor_slices((X_validation, validation.label.to_numpy())).batch(30).with_options(options)
history = model.fit(train_data, validation_data=val_data, epochs=int(os.getenv('EPOCHS','10')),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], shuffle=False, verbose=2)

# 5. Convert logits to probabilities, then evaluate the untouched test split.
logits = []
for start in range(0,len(X_test),64):
    logits.extend(model(X_test[start:start+64],training=False).numpy().ravel())
probabilities = tf.math.sigmoid(np.array(logits)).numpy()
predictions = (probabilities>=.5).astype(int)
report = {'accuracy':float(accuracy_score(test.label,predictions)),
          'f1_macro':float(f1_score(test.label,predictions,average='macro')),
          'confusion':confusion_matrix(test.label,predictions,labels=[0,1]).tolist(),
          'train_size':len(train),'validation_size':len(validation),'test_size':len(test),
          'sample_limit':samples,'epochs':len(history.history['loss']),'history':history.history,
          'labels':['Fake dataset label','Real dataset label']}
model.save(out/'news.keras')
(out/'news_tokenizer.json').write_text(tokenizer.to_json())
(out/'news_report.json').write_text(json.dumps(report,indent=2))
test[['title','text','label']].head(10).to_csv(out/'news_examples.csv',index=False)
print('News test accuracy:',report['accuracy'],flush=True)
