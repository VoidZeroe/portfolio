# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:33:51 2021

@author: hp
"""

import tensorflow as tf
# tf.enable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import tensorflow_datasets as tfds
from official.nlp.bert import tokenization
import official.nlp as nlp
import official.nlp.optimization



# tf.config.run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)
# config = tf.ConfigProto(device_count={"CPU": 2})
# tf.keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    # each sentence should end with a [SEP] (separator) token
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(glue_dict, tokenizer):
    # start by encoding all the sentences and packing them into ragged-tensors
    sentence1 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(glue_dict["sentence2"])])
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    # now prepend a [CLS] token, and concatenate the ragged tensors 
    # to form a single input_word_ids tensor for each example
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
    # the mask allows the model to cleanly differentiate between the content and the padding
    # it has the same shape as the input_word_ids, and contains a 1 anywhere the input_word_ids 
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    # input_type_ids tensor also has the same shape, but inside the non-padded region 
    # it contains a 0 or a 1 indicating which sentence the token is a part of
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()
    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}
    return inputs
max_seq_length = 103
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
bert_inputs = {'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
pooled_output, _ = bert_layer([input_word_ids, input_mask, input_type_ids])
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
output = tf.keras.layers.Dropout(rate=0.2)(pooled_output)
initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
bert_output = tf.keras.layers.Dense(2, kernel_initializer=initializer, name='output')(output)
model = tf.keras.models.Model(inputs=bert_inputs, outputs=bert_output)
model.summary()

glue, info = tfds.load('glue/mrpc', with_info=True, batch_size=-1)
glue_train = bert_encode(glue['train'], tokenizer)
glue_train_labels = glue['train']['label']
glue_validation = bert_encode(glue['validation'], tokenizer)
glue_validation_labels = glue['validation']['label']
epochs = 10
batch_size = 32
eval_batch_size = 32
train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(
    glue_train, glue_train_labels,
    validation_data=(glue_validation, glue_validation_labels),
    batch_size=batch_size,
    validation_batch_size=eval_batch_size,
    epochs=epochs)

my_examples = bert_encode(
    glue_dict={
        'sentence1': [
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2': [
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    },
    tokenizer=tokenizer)
result = model.predict(my_examples)
print(result)
result = tf.argmax(result).numpy()
print(result)
print(np.array(info.features['label'].names)[result])
