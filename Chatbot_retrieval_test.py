import tensorflow as tf
import numpy as np
import pickle
import helpers
import sys

pickle_in = open("ProcessedData.pickle","rb")

data = pickle.load(pickle_in)
train_questions = data['train_questions']
train_answers = data['train_answers']
train_targets = data['train_targets']
valid_questions = data['valid_questions']
valid_answers = data['valid_answers']
valid_targets = data['valid_targets']
queW2int = data['queW2int']
queint2W = data['queint2W']
ansW2int = data['ansW2int']
ansint2W = data['ansint2W']

#parameters
encoder_hidden_units = 50
vocab_size = len(queint2W) + 100
input_embedding_size = 50
rnn_dim = 50
batch_size = 1

# mode = 'train'
# mode = 'infer'

#data
context = tf.placeholder(shape=(None,None), dtype=tf.int32, name='context')
utterance = tf.placeholder(shape=(None,None), dtype=tf.int32, name='utterance')
concatInputs = tf.placeholder(shape=(None,None), dtype=tf.int32, name='utterance')
targets = tf.placeholder(shape=(None,None), dtype=tf.int32, name='targets')
context_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='context_length')
utterance_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='utterance_length')
concatInputs_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='concatInputs_length')

#embeddings

# Initialize embedidngs randomly or with pre-trained vectors if available
initializer = tf.random_uniform_initializer(-0.25, 0.25)
embeddings = tf.get_variable("word_embeddings",shape=[vocab_size, input_embedding_size],initializer=initializer)

# embedded_concat = tf.nn.embedding_lookup(embeddings, concatInputs, name="embed_context")

# Embed the context and the utterance
embedded_context = tf.nn.embedding_lookup(embeddings, context, name="embed_context")
embedded_utterance = tf.nn.embedding_lookup(embeddings, utterance, name="embed_utterance")
  

cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

# Run the utterance and context through the RNN

inputs = tf.concat([embedded_context, embedded_utterance], 0)
sequence_length_ = tf.concat([context_len, utterance_len], 0)

print(inputs.get_shape())
outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length = None, dtype=tf.float32)
encoded_context, encoded_utterance = tf.split(states.h, 2, 0)

# with tf.variable_scope('forward'):
#     outputsC, encoded_context = tf.nn.dynamic_rnn(cell, embedded_context, sequence_length = context_len, dtype=tf.float32)
# 
# with tf.variable_scope('backword'):
#     outputsU, encoded_utterance = tf.nn.dynamic_rnn(cell, embedded_utterance, sequence_length = utterance_len, dtype=tf.float32)

#prediction
M = tf.get_variable("M",
                    shape=[rnn_dim, rnn_dim],
                    initializer=tf.truncated_normal_initializer())

encoder_response = tf.matmul(encoded_context, M, True)
encoder_response = tf.expand_dims(encoder_response, 2)
encoded_utterance = tf.expand_dims(encoded_utterance, 2)

# Dot product between generated response and actual response
# (c * M) * r
logits = tf.matmul(encoder_response, encoded_utterance, True)
logits = tf.squeeze(logits, [1])

# Apply sigmoid to convert logits to probabilities
probabilities = tf.sigmoid(logits)

# if mode == 'infer':
#   return probabilities, None

# Calculate the binary cross-entropy loss
# print(logits.get_shape())
# targets = np.reshape(targets, (len(targets), 1))
# labels = (tf.to_float(targets))
# print(labels.get_shape())
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.to_float(targets), logits = logits)

# Mean loss across the batch of examples
loss = tf.reduce_mean(losses, name="mean_loss")
train_op = tf.train.AdamOptimizer().minimize(loss)
# return probabilities, loss_mean

# sess.run(tf.global_variables_initializer())

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

def batch_data(questions, answers, train_targets, batch_size):
    batchRange = len(questions)//batch_size
    print(batch_size)
    
    for batch_i in range(0, batchRange):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        targets_batch = train_targets[start_i:start_i + batch_size]
        concat_batch = questions_batch + answers_batch
        yield questions_batch, answers_batch, targets_batch, concat_batch

batches = batch_data(train_questions, train_answers, train_targets, batch_size)

def next_feed():
    que_batch, ans_batch, targets_batch, concat_batch = next(batches)
    context_, context_len_ = helpers.batch(que_batch)
    utterance_, utterance_len_ = helpers.batch(ans_batch)
    concatInputs_, concatInputs_len_ = helpers.batch(concat_batch)
    targets_, _ = helpers.batch(targets_batch)
            
    return {
        context : context_,
        utterance : utterance_,
        concatInputs : concatInputs_,
        concatInputs_len : concatInputs_len_,
        targets : targets_,
        context_len : context_len_,
        utterance_len : utterance_len_,
    }


# loss_track = []
# batches_in_epoch = 128

saver = tf.train.Saver()

with tf.Session() as sess:
    
    fd = next_feed()
#     print(len(fd[context]))
#     print(len(fd[utterance]))
    saver.restore(sess, "C:/MyStuff/SEM3/DL/Project1/chatbot-retrieval-master/output/model.ckpt")
    print("Model restored.")
    
    user_question = ''
    while(user_question != 'q'):
        bestAnswerIndex = 0
        bestScore = -sys.maxsize - 1
        user_question = input("Ask me question: ")
        user_que_words = user_question.split()
        my_question = []
        for w in user_que_words:
            if w in queW2int:
                my_question.append(queW2int[w])
            else:
                queW2int[w] = len(queW2int)
                my_question.append(queW2int[w])
        for i in range(0,len(fd[context][:,0]) - len(my_question)):
            my_question.append(0)
        fd[context][:,0] = my_question
        answerIndex = 0
        for my_answers in train_answers[:20]:
            fd[utterance][:,0] = my_answers
            predict_ = sess.run(probabilities, fd)
            
#             currentScore = np.sum(predict_)
            currentScores = predict_
            currentScore = 0
            for wordcntr in range(len(my_answers)):
                if my_answers[wordcntr] != 0:
#                     print(predict_[wordcntr,0])
                    currentScore += predict_[wordcntr,0]*1000.0
            
            if currentScore > bestScore:
                bestAnswerIndex = answerIndex
                bestScore = currentScore
            answerIndex += 1
            print('current Score ', currentScore)
        print('best Score ', bestScore)
        bestAnswer = train_answers[bestAnswerIndex]
        print('    predicted > {}'.format([ansint2W[i] for i in bestAnswer if i in ansint2W and i != 0]))