import tensorflow as tf
# 所有参数
RNN_Cell = tf.nn.rnn_cell.LSTMCell
hidden_size = 450
batch_size = 128
cell_nums = 2
epoch_num = 10
lr = 0.001
clip = 5.0
dropout = 0.75
num_tags = 8
update_embedding = True
embedding_dim = 200
shuffle = True
isTrain = True
CRF = True
