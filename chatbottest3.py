
import numpy as np
import tensorflow as tf
import helpers
import processedData as pd

print(pd.train_questions[0])
tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1
batch_size = 100
vocab_size = len(pd.ansv2int)
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

#embeddings
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )
  
  
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

decoder_cell = LSTMCell(decoder_hidden_units)

# encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = encoder_inputs_length + 2

W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)
    
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)
    
def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2, name = 'decoder_prediction')

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

def batch_data(questions, answers, batch_size):
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        yield questions_batch, answers_batch

batches = batch_data(pd.train_questions, pd.train_answers, batch_size)
# batches = helpers.random_sequences(length_from=3, length_to=8,
#                                    vocab_lower=2, vocab_upper=10,
#                                    batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)
print()
for seq in next(batches)[:10]:
    print(seq)
#
# def next_feed():
#     batch = next(batches)
#     encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
#     decoder_targets_, _ = helpers.batch([(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
#     )
#     return {
#         encoder_inputs: encoder_inputs_,
#         encoder_inputs_length: encoder_input_lengths_,
#         decoder_targets: decoder_targets_,
#     }
         
def next_feed():
    que_batch, ans_batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(que_batch)
    decoder_targets_, decoder_input_lengths_ = helpers.batch([(sequence) + [EOS] + [PAD] * 2 for sequence in ans_batch])
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }
    
loss_track = []
max_batches = 30001
batches_in_epoch = 1000

train = True

if train:
	saver = tf.train.Saver()
	try:
		for batch in range(max_batches):
			print('batch number : ', batch)
			fd = next_feed()
			fd[decoder_targets] = fd[decoder_targets][:(len(fd[encoder_inputs]) + 2),:]
			print(type(fd))
			print(type(fd[encoder_inputs]))
			print(len(fd[encoder_inputs]))
			print(len(fd[decoder_targets]))
    #             print(len(fd[encoder_inputs][0]))
    #             print(len(fd[decoder_targets][0]))
			_, l = sess.run([train_op, loss], fd)
			loss_track.append(l)
    
			if batch%batches_in_epoch == 0:#batch == 0 or batch % batches_in_epoch == 0:
				print('batch {}'.format(batch))
				print('  minibatch loss: {}'.format(sess.run(loss, fd)))
				print("question 1: ", fd[encoder_inputs][:,0])
				user_question = ''
				while(user_question != 'q'):
					user_question = input("Ask me question: ")
					user_que_words = user_question.split()
					my_question = []
					for w in user_que_words:
						if w in pd.quev2int:
							my_question.append(pd.quev2int[w])
						else:
							pd.quev2int[w] = 8095
							my_question.append(8095)
					for i in range(0,len(fd[encoder_inputs][:,0]) - len(my_question)):
						my_question.append(0)
					fd[encoder_inputs][:,0] = my_question
			#                 my_question_batch = [[2265, 603, 7350, 8094]]
			#                 encoder_inputs_, encoder_input_lengths_ = helpers.batch(my_question_batch)
			#                 my_question = {encoder_inputs: encoder_inputs_, encoder_inputs_length: encoder_input_lengths_,}
			#                 my_question = {encoder_inputs : [[2265, 603, 7350, 8094]], encoder_inputs_length : [4]}
					predict_ = sess.run(decoder_prediction, fd)
					saver.save(sess, 'C:/MyStuff/SEM3/DL/Project1/ChatBot/saved/my-model.ckpt', global_step = 1003)
					print("saved")
					for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
						print('  sample {}:'.format(i + 1))
						print('    input     > {}'.format([pd.queint2v[i] for i in inp if i != 0]))
						print('    predicted > {}'.format([pd.queint2v[i] for i in pred if i != 0]))
						if i == 0:
							break
					print()
    
	except KeyboardInterrupt:
		print('training interrupted')
else:
	sess.close()
	print('testing')
    
	sess=tf.Session()
	new_saver = tf.train.import_meta_graph('C:/MyStuff/SEM3/DL/Project1/ChatBot/saved/my-model.ckpt-1002.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('C:/MyStuff/SEM3/DL/Project1/ChatBot/saved/'))
	fd = next_feed() 
	graph = tf.get_default_graph() 
    
#     decoder_prediction = graph.get_tensor_by_name("decoder_prediction:0")
	predict_ = sess.run(decoder_outputs_ta, fd)
    
    
	for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
		print('  sample {}:'.format(i + 1))
		print('    input     > {}'.format(inp))
		print('    predicted > {}'.format(pred))
		if i >= 2:
			break
	print()