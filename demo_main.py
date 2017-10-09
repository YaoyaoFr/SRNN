import tensorflow as tf
from demo_rnn import RNNStructure

with tf.Session() as sess:
    rnn_structure = RNNStructure(sess=sess)

    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.global_variables_initializer()
    epoch = 0

    # Initialization
    sess.run(init_op)
    print('Initialize successful.')

    is_train = True

    # Data
    input_data = None
    output_data = None
    hidden_state = None

    while True:
        # Test Data Initialization

        # Training
        if is_train:
            train_data = rnn_structure.train_time_sequence(input_data=input_data, output_data=output_data, state=hidden_state)

