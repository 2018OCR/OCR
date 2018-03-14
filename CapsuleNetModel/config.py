import os
import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')

# For spread loss
flags.DEFINE_float('m_scheduler', 1, '.')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.0005, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')
flags.DEFINE_float('epsilon', 1e-9, 'void NAN')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'mnist', 'The name of dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
# flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
results = os.path.join('.', 'results')
logdir = os.path.join(results, 'logdir')
flags.DEFINE_string('results', results, 'path for saving results')
flags.DEFINE_string('logdir', logdir, 'logs directory')
flags.DEFINE_string('model', 'CapsNetNrePlus', 'the model to use')
flags.DEFINE_boolean('debug', True, 'debug mode')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 1, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 16, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 8, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)