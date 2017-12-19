
# coding: utf-8

# # Building a Chatbot

# In this project, we will build a chatbot using conversations from Cornell University's [Movie Dialogue Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The main features of our model are LSTM cells, a bidirectional dynamic RNN, and decoders with attention. 
# 
# The conversations will be cleaned rather extensively to help the model to produce better responses. As part of the cleaning process, punctuation will be removed, rare words will be replaced with "UNK" (our "unknown" token), longer sentences will not be used, and all letters will be in the lowercase. 
# 
# With a larger amount of data, it would be more practical to keep features, such as punctuation. However, I am using FloydHub's GPU services and I don't want to get carried away with too training for too long.

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time

print("preprocessing...")
movieL = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
movieC = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

idToLine = {}
for l in movieL:
    w = l.split(' +++$+++ ')
    if len(w) == 5:
        idToLine[w[0]] = w[4]

conversations = [ ]
for l in movieC[:-1]:
    w = l.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations.append(w.split(','))

context = []
responses = []

for conversation in conversations:
    for i in range(len(conversation)-1):
        context.append(idToLine[conversation[i]])
        responses.append(idToLine[conversation[i+1]])


replacements = open('replacements.txt', 'r').read().split('\n')

def replaceText(txt):

    txt = txt.lower()
    
    for replace in replacements:
        replacement = replace.split(',')
        txt = re.sub(replacement[0], replacement[1], txt)
    
    txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", txt)
    return txt

replacedQues = []
for que in context:
    replacedQues.append(replaceText(que))
    
replacedAns = []    
for ans in responses:
    replacedAns.append(replaceText(ans))

queTemp = []
ansTemp = []

i = 0
for que in replacedQues:
    if len(que.split()) >= 2 and len(que.split()) <= 20:
        queTemp.append(que)
        ansTemp.append(replacedAns[i])
    i += 1

smallQue = []
smallAns = []

i = 0
for ans in ansTemp:
    if len(ans.split()) >= 2 and len(ans.split()) <= 20:
        smallAns.append(ans)
        smallQue.append(queTemp[i])
    i += 1

vocabulary = {}
for que in smallQue:
    for w in que.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1
            
for ans in smallAns:
    for w in ans.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1


count = 0
for k,v in vocabulary.items():
    if v >= 10:
        count += 1

quev2int = {}

word_num = 0
for word, count in vocabulary.items():
    if count >= 10:
        quev2int[word] = word_num
        word_num += 1
        
ansv2int = {}

word_num = 0
for word, count in vocabulary.items():
    if count >= 10:
        ansv2int[word] = word_num
        word_num += 1

wilds = ['<PAD>','<EOS>','<UNK>','<GO>']

for wild in wilds:
    quev2int[wild] = len(quev2int)+1
    
for wild in wilds:
    ansv2int[wild] = len(ansv2int)+1

queint2v = {v_i: v for v, v_i in quev2int.items()}
ansint2v = {v_i: v for v, v_i in ansv2int.items()}

for i in range(len(smallAns)):
    smallAns[i] += ' <EOS>'

queInt = []
for que in smallQue:
    ints = []
    for w in que.split():
        if w not in quev2int:
            ints.append(quev2int['<UNK>'])
        else:
            ints.append(quev2int[w])
    queInt.append(ints)
    
ansInt = []
for ans in smallAns:
    ints = []
    for w in ans.split():
        if w not in ansv2int:
            ints.append(ansv2int['<UNK>'])
        else:
            ints.append(ansv2int[w])
    ansInt.append(ints)

def model_inputs():
    '''Create palceholders for inputs to the model'''
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob


# In[32]:

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[33]:

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    '''Create the encoding layer'''
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs, 
                                                   dtype=tf.float32)
    return enc_state


# In[34]:

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    '''Decode the training data'''
    
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    
    att_keys, att_vals, att_score_fn, att_construct_fn =             tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    
    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name = "attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                              train_decoder_fn, 
                                                              dec_embed_input, 
                                                              sequence_length, 
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)


# In[35]:

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    '''Decode the prediction data'''
    
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    
    att_keys, att_vals, att_score_fn, att_construct_fn =             tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn, 
                                                                         encoder_state[0], 
                                                                         att_keys, 
                                                                         att_vals, 
                                                                         att_score_fn, 
                                                                         att_construct_fn, 
                                                                         dec_embeddings,
                                                                         start_of_sequence_id, 
                                                                         end_of_sequence_id, 
                                                                         maximum_length, 
                                                                         vocab_size, 
                                                                         name = "attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                                infer_decoder_fn, 
                                                                scope=decoding_scope)
    
    return infer_logits


# In[36]:

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    '''Create the decoding cell and input the parameters for the training and inference decoding layers'''
    
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                vocab_size, 
                                                                None, 
                                                                scope=decoding_scope,
                                                                weights_initializer = weights,
                                                                biases_initializer = biases)

        train_logits = decoding_layer_train(encoder_state, 
                                            dec_cell, 
                                            dec_embed_input, 
                                            sequence_length, 
                                            decoding_scope, 
                                            output_fn, 
                                            keep_prob, 
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state, 
                                            dec_cell, 
                                            dec_embeddings, 
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'], 
                                            sequence_length - 1, 
                                            vocab_size,
                                            decoding_scope, 
                                            output_fn, keep_prob, 
                                            batch_size)

    return train_logits, infer_logits


# In[37]:

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, 
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, 
                  questions_vocab_to_int):
    
    '''Use the previous functions to create the training and inference logits'''
    
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, 
                                                       answers_vocab_size+1, 
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    train_logits, infer_logits = decoding_layer(dec_embed_input, 
                                                dec_embeddings, 
                                                enc_state, 
                                                questions_vocab_size, 
                                                sequence_length, 
                                                rnn_size, 
                                                num_layers, 
                                                questions_vocab_to_int, 
                                                keep_prob, 
                                                batch_size)
    return train_logits, infer_logits


# In[38]:

# Set the Hyperparameters
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75


# In[39]:

# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
# Start the session
sess = tf.InteractiveSession()
    
# Load the model inputs    
input_data, targets, lr, keep_prob = model_inputs()
# Sequence length will be the max line length for each batch
sequence_length = tf.placeholder_with_default(max_line_length, None, name='sequence_length')
# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create the training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(answers_vocab_to_int), 
    len(questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, 
    questions_vocab_to_int)

# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
logits = tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# In[40]:

def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[41]:

def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        yield pad_questions_batch, pad_answers_batch


# In[42]:

# Validate the training with 10% of the data
train_valid_split = int(len(sorted_questions)*0.15)

# Split the questions and answers into training and validating data
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_answers[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))


# In[43]:

display_step = 100 # Check training loss after every 100 batches
stop_early = 0 
stop = 5 # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = ((len(train_questions))//batch_size//2)-1 # Modulus for checking validation loss
total_train_loss = 0 # Record the training loss for each display step
summary_valid_loss = [] # Record the validation loss for saving improvements in the model

checkpoint = "best_model.ckpt" 

def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''
    
    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

#sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    saver.restore(sess, 'D:/DeepLearning/data/final/best_model.ckpt')
    random = np.random.choice(len(valid_questions))
    #input_question = short_questions[random]
    input_question = 'I saw her yesterday while I was on my to gym and said hew Hello'


	# Prepare the question
    input_question = question_to_seq(input_question, questions_vocab_to_int)

	# Pad the questions until it equals the max_line_length
    input_question = input_question + [questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))
	# Add empty questions so the the input_data is the correct shape
    batch_shell = np.zeros((batch_size, max_line_length))
	# Set the first question to be out input question
    batch_shell[0] = input_question    
		
	# Run the model with the input question
    answer_logits = sess.run(logits, {input_data: batch_shell, 
												keep_prob: 1.0})[0]

	# Remove the padding from the Question and Answer
    pad_q = questions_vocab_to_int["<PAD>"]
    pad_a = answers_vocab_to_int["<PAD>"]
    
    print('Question')
    print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
    print('  Input Words: {}'.format([questions_int_to_vocab[i] for i in input_question if i != pad_q]))
    
    print('\nAnswer')
    print('  Word Ids:      {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
    print('  Response Words: {}'.format([answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))






