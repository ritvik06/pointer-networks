"""Implementation of Pointer networks: http://arxiv.org/pdf/1506.03134v1.pdf.
"""


from __future__ import absolute_import, division, print_function

import random

import math
import numpy as np
import tensorflow as tf

from dataset import DataGenerator
from pointer import pointer_decoder

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, 'Batch size.  ')
flags.DEFINE_integer('max_len', 50, 'Size of problem.')
flags.DEFINE_integer('rnn_size', 512, 'Number of RNN cells in each layer')
flags.DEFINE_integer('num_layers', 1, 'Number of layers in the network.')
flags.DEFINE_string('problem_type', 'convex_hull', 'What kind of problem to train on: "convex_hull", or "sort".')
flags.DEFINE_string('pointer_type', 'one_hot', 'What kind of pointer to use: "multi_hot", "one_hot", or "soft_max"')
flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')
flags.DEFINE_float("max_gradient_norm", 2.0, "Clip gradients to this norm.")
flags.DEFINE_float('learning_rate', 1.0, "Learning rate.")
FLAGS = flags.FLAGS

class PointerNetwork(object):
    
    def __init__(self, max_len, input_size, size, num_layers, max_gradient_norm, batch_size, learning_rate):
        """Create the network.
        
        Args:
            max_len: maximum length of the model.
            input_size: size of the inputs data.
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
        """
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)

        
        cell = tf.nn.rnn_cell.LSTMCell(num_units=size, initializer=tf.contrib.layers.xavier_initializer())
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
            
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []
        for i in range(max_len):
            self.encoder_inputs.append(tf.placeholder(
                tf.float32, [batch_size, input_size], name="EncoderInput%d" % i))

        for i in range(max_len + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.float32, [batch_size, input_size], name="DecoderInput%d" % i))
            self.decoder_targets.append(tf.placeholder(
                tf.float32, [batch_size, max_len + 1], name="DecoderTarget%d" % i))  # one hot
            self.target_weights.append(tf.placeholder(
                tf.float32, [batch_size, 1], name="TargetWeight%d" % i))
        
        # Need for attention
        encoder_outputs, final_state = tf.nn.rnn(cell, self.encoder_inputs, dtype = tf.float32)
        
        # Need a dummy output to point on it. End of decoding.
        encoder_outputs = [tf.zeros([batch_size, size])] + encoder_outputs

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)

        with tf.variable_scope("decoder"):
            outputs, states, _ = pointer_decoder(
                self.decoder_inputs, final_state, attention_states, cell, feed_prev=False, pointer_type=FLAGS.pointer_type)

        with tf.variable_scope("decoder", reuse=True):
            predictions, _, inps = pointer_decoder(
                self.decoder_inputs, final_state, attention_states, cell, feed_prev=True, pointer_type=FLAGS.pointer_type)
            
        self.predictions = predictions

        self.outputs = outputs
        self.inps = inps
        
            
    def create_feed_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        feed_dict = {}
        for placeholder, data in zip(self.encoder_inputs, encoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_inputs, decoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_targets, decoder_target_data):
            feed_dict[placeholder] = data

        for placeholder in self.target_weights:
            feed_dict[placeholder] = np.ones([self.batch_size, 1])

        return feed_dict

    def step(self):

        loss = 0.0
        # for output, target, weight in zip(self.outputs, self.decoder_targets, self.target_weights):
        for output, target, weight in zip(self.predictions, self.decoder_targets, self.target_weights):
            loss += tf.nn.softmax_cross_entropy_with_logits(output, target) * weight

        loss = tf.reduce_mean(loss)

        # test_loss = 0.0
        # for output, target, weight in zip(self.predictions, self.decoder_targets, self.target_weights):
        #     test_loss += tf.nn.softmax_cross_entropy_with_logits(output, target) * weight
        
        # test_loss = tf.reduce_mean(test_loss)
        # tf.scalar_summary('test_loss', test_loss)

        train_op = tf.contrib.layers.optimize_loss(
            loss, self.global_step, learning_rate=self.learning_rate,
            optimizer="SGD", clip_gradients=self.max_gradient_norm, name="train_op")
        
        train_loss_value = 0.0
        test_loss_value = 0.0
        
        correct_order = 0.0
        all_order = 0.0

        sess = tf.Session()
        with sess.as_default():
            previous_losses = []
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter("./tmp/pointer_logs/"+ FLAGS.problem_type +"/" + FLAGS.pointer_type+ "/train", sess.graph)
            test_writer = tf.train.SummaryWriter("./tmp/pointer_logs/"+ FLAGS.problem_type +"/" + FLAGS.pointer_type + "/test", sess.graph)
            init = tf.initialize_all_variables()
            sess.run(init)
            for i in range(int(math.ceil(1000000/self.batch_size))):
                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    self.batch_size, self.max_len, convex_hull=(FLAGS.problem_type=="convex_hull"))
                # Train
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)
                if (i+1)%10 == 0:
                    if (i+1)%FLAGS.steps_per_checkpoint == 0:
                        #record run metadata
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, d_x, l = sess.run([merged, loss, train_op], feed_dict=feed_dict, 
                                options=run_options,
                                run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d'%(i+1))
                        train_writer.add_summary(summary, (i+1))

                    else:
                        summary, d_x, l = sess.run([merged, loss, train_op], feed_dict=feed_dict)
                        train_writer.add_summary(summary, (i+1))
                else:
                    d_x, l = sess.run([loss, train_op], feed_dict = feed_dict)

                train_loss_value += d_x/FLAGS.steps_per_checkpoint

                if (i+1) % FLAGS.steps_per_checkpoint == 0:
                    print('Step:', i+1, 'Learning rate:', self.learning_rate)
                    print("Train: ", train_loss_value)
                    train_loss_value = 0

                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    self.batch_size, self.max_len, train_mode=False, convex_hull=(FLAGS.problem_type=="convex_hull"))
                # Test
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)
                inps_ = sess.run(self.inps, feed_dict=feed_dict)
                predictions = sess.run(self.predictions, feed_dict=feed_dict)
                
                
                
                if (i+1)%10 == 0:
                    summary, test_loss_ = sess.run([merged, loss], feed_dict=feed_dict)
                    test_writer.add_summary(summary, (i+1))
                else:
                    test_loss_ = sess.run(loss, feed_dict=feed_dict)
                test_loss_value += test_loss_/FLAGS.steps_per_checkpoint

                if (i+1) % FLAGS.steps_per_checkpoint == 0:
                    print("Test: ", test_loss_value)
                    test_loss_value = 0

                predictions_order = np.concatenate([np.expand_dims(prediction , 0) for prediction in predictions])
                predictions_order = np.argmax(predictions_order, 2).transpose(1, 0)[:,0:self.max_len]
                
                targets_order = np.concatenate([np.expand_dims(target, 0) for target in targets_data])
                targets_order = np.argmax(targets_order,2).transpose(1,0)[:,0:self.max_len]

                # for j in xrange(self.batch_size):
                #         target_order = targets_order[j]
                #         num_vertices = np.where(target_order==0)[0][0]
                #         pred_order = predictions_order[j]
                #         pred_sequence = predicted_order[0:num_vertices]
                #         pred_order[0:num_vertices] = np.roll(pred_sequence, -list(pred_sequence).index(np.min(pred_sequence)))
                #         correct_order += (pred_order == target_order)
                
                correct_order += np.sum(np.all(predictions_order == targets_order, axis=1))
                all_order += self.batch_size

                if (i+1) % 100 == 0:
                    print('Correct order / All order: %f' % (correct_order / all_order))
                    correct_order = 0
                    all_order = 0
                    print("----")

            #evaluate the trained network



if __name__ == "__main__":
    pointer_network = PointerNetwork(FLAGS.max_len, 2 - (FLAGS.problem_type == 'sort'), FLAGS.rnn_size,
                                     FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size, FLAGS.learning_rate)
    dataset = DataGenerator()
    pointer_network.step()

