import tensorflow as tf
# 所有参数
RNN_Cell = tf.nn.rnn_cell.LSTMCell
hidden_size = 450
batch_size = 64
cell_nums = 2
epoch_num = 10
lr = 0.001
clip = 5.0
dropout = 0.75
num_tags = 16
update_embedding = True
embedding_dim = 200
shuffle = False
isTrain = False
CRF = True
max_model_num = 5
