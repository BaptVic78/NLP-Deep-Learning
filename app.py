"""Visualize the three existing notebook models, using prepared local files."""
from pathlib import Path
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from text_cleaning import normalize_text

root = Path(__file__).resolve().parent
out = root/'artifacts'
st.set_page_config(page_title='RNN / LSTM lab',layout='wide')
st.title('RNN / LSTM lab')
st.write('Two notebook exercises, with their inputs, predictions and measured results.')
task = st.sidebar.radio('Exercise',['Fake-news dataset','Emotions'])
key = 'news' if task == 'Fake-news dataset' else 'emotions'
tabs = st.tabs(['Demonstration','Evaluation','How it works'])

@st.cache_resource
def load_text_model(name):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    model = tf.keras.models.load_model(out/f'{name}.keras',compile=False)
    tokenizer = tokenizer_from_json((out/f'{name}_tokenizer.json').read_text())
    return model,tokenizer

report_path = out/f'{key}_report.json'
report = json.loads(report_path.read_text()) if report_path.exists() else None

with tabs[0]:
    if key=='news':
        st.subheader('Classify an English headline + article')
        title = st.text_input('Headline','')
        text = st.text_area('Article text','',height=160)
        raw = title+' '+text
        st.caption('Dataset classification, not a fact-checking service. A model score does not verify a claim.')
    else:
        st.subheader('Classify the emotion of an English comment')
        raw = st.text_area('Comment','I am happy to see my friends again.',height=140)
    examples_path = out/f'{key}_examples.csv'
    if examples_path.exists():
        with st.expander('Held-out examples from the test set'):
            examples = pd.read_csv(examples_path)
            st.dataframe(examples,hide_index=True)
            st.caption('Examples are from the test set; their labels are not used as predictions.')
    if st.button('Predict',type='primary'):
        cleaned = normalize_text(raw,remove_numbers=key=='emotions')
        if not cleaned:
            st.warning('Enter some English text.')
        elif not report:
            st.info(f'Train this exercise first: venv/bin/python train_{key}.py')
        else:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            with st.spinner('Loading the notebook model…'):
                model,tokenizer = load_text_model(key)
                tokens = tokenizer.texts_to_sequences([cleaned])
                sequence = pad_sequences(tokens,maxlen=256,padding='post')
                output = model(sequence,training=False).numpy()[0]
            if not tokens[0]:
                st.warning('No words are recognized by the training vocabulary.')
            else:
                if key=='news':
                    probability = float(tf.math.sigmoid(output[0]).numpy())
                    probabilities = [1-probability,probability]
                else:
                    probabilities = output.tolist()
                label = report['labels'][int(np.argmax(probabilities))]
                st.success(f'Prediction: {label}')
                scores = pd.DataFrame({'Class':report['labels'],'Model probability':probabilities})
                st.plotly_chart(px.bar(scores,x='Class',y='Model probability',range_y=[0,1]),width='stretch')
                st.caption('Model probabilities are not calibrated guarantees.')
                with st.expander('See the preprocessing'):
                    st.code(cleaned,language=None)
                    st.write(f'{len(tokens[0])} recognized tokens. Padding/truncation produces 256 positions.')

with tabs[1]:
    if report:
        st.caption(f"Prepared run: {report['epochs']} epochs · {report['train_size']} training examples · {report['test_size']} test examples.")
        if report.get('sample_limit'):
            st.info(f"Quick demonstration run with SAMPLES={report['sample_limit']}. These results are not the full notebook training run.")
        a,b=st.columns(2)
        a.metric('Test accuracy',f"{report['accuracy']:.1%}")
        b.metric('Test macro F1',f"{report['f1_macro']:.1%}")
        matrix = pd.DataFrame(report['confusion'],index=report['labels'],columns=report['labels'])
        st.plotly_chart(px.imshow(matrix,text_auto=True,color_continuous_scale='Teal',labels={'x':'Predicted','y':'Actual'}),width='stretch')
        history = pd.DataFrame(report['history'])
        history.index = history.index+1
        st.subheader('Actual training history')
        st.line_chart(history[[name for name in ['loss','val_loss'] if name in history]])
        if 'accuracy' in history:
            st.line_chart(history[['accuracy','val_accuracy']])
    else:
        st.info('Evaluation appears after a completed training run. No metrics are fabricated.')

with tabs[2]:
    if key=='news':
        st.code('Headline + text → normalization → tokenizer → 256 tokens → Embedding(128) → BiLSTM(64) → BiLSTM(16) → Dense → logit',language=None)
        st.write('Sigmoid converts the final logit to a score between 0 and 1. The classification boundary is probability 0.5, equivalent to logit 0. Vocabulary is learned on train only.')
    else:
        st.code('Comment → normalization → tokenizer → 256 tokens → Embedding(128) → BiLSTM(64) → BiLSTM(32) → dropout → softmax(6)',language=None)
        st.write('Softmax gives six class scores. The predicted class has the largest score. The official validation split is used for early stopping; the test split is used for final evaluation.')
    st.markdown('**LSTM is a type of recurrent neural network (RNN).** Bidirectional means reading a text sequence in both directions; it is used only for the classification exercises here.')
    st.write('The original notebooks are preserved. Each training script follows their architecture and saves the model, tokenizer and actual evaluation for this interface.')
