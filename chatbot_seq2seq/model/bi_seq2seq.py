import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import pickle
import data_unit as du
import random

# Learning rate
lr = 0.001
# Mini-batch size
batch_size = 128
# Number of cells in each layer
size_layer = 256
# Number of layers
num_layers = 2
# Embedding size for encoding part and decoding part
embedding_size = 300
# Number of max epochs for training
epochs = 20
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = 1
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 1


# 读取数据
# 读取数据
train_data,  word2id, idx2word = du.read_train_data("question_answer.pk")
# print(len(train_data), train_data[:3])
# print(word2id["，"])
# print(idx2word[2498])

# 数据转化为id表示
train_data_int = du.map_int(train_data, word2id)
# test_data_int = du.map_int(test_data, word2id)
# print(train_data_int[:3])
# 补齐数据为bath_size整数倍
train_remainder = len(train_data_int) % batch_size
# test_remainder = len(train_data_int) % batch_size

# 训练集
train_source = train_data_int + train_data_int[0:batch_size-train_remainder]
# 测试集
# test_source = test_data_int + test_data_int[0:batch_size-test_remainder]

# 词汇总数
vocab_size = len(word2id)


# 对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
def pad_sentence_batch(sentence_batch, pad_int):
    # max_sentence = max([len(sentence) for sentence in sentence_batch])
    # return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    source_int_max_sentence = max([len(sentence[0]) for sentence in sentence_batch])
    target_int_max_sentence = max([len(sentence[1]) for sentence in sentence_batch])
    source = [sentence[0] + [pad_int] * (source_int_max_sentence - len(sentence[0])) for sentence in sentence_batch]
    target = [sentence[1] + [pad_int] * (target_int_max_sentence - len(sentence[1])) for sentence in sentence_batch]
    return source, target


def get_batches(data, pad_int):
    """
     # 定义生成器，用来获取batch
    :param data:
    :param pad_int:
    :return:
    """
    for batch_i in range(0, len(data) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = data[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch, pad_targets_batch = pad_sentence_batch(sources_batch, pad_int)
        # 记录每条记录的长度
        source_lengths = []
        for source in pad_sources_batch:
            source_lengths.append(len(source))

        targets_lengths = []
        for target in pad_targets_batch:
            targets_lengths.append(len(target))
        yield np.array(pad_targets_batch), np.array(pad_sources_batch), targets_lengths, source_lengths


def lstm_cell(size_layer,dropout_keep_prob, reuse=False):
    """
     构建一个单独的 RNNCell
    :param size_layer:
    :param dropout_keep_prob:
    :param reuse:
    :return:
    """
    lstm_cell = tf.nn.rnn_cell.LSTMCell(size_layer, reuse=reuse)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell,
        dtype=tf.float32,
        output_keep_prob=dropout_keep_prob,
        )
    return cell


def attention(encoder_out, seq_len, keep_prob, reuse=False):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=size_layer,
                                                               memory=encoder_out,
                                                               memory_sequence_length=seq_len)
    attention_out = tf.contrib.seq2seq.AttentionWrapper(cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell(keep_prob,reuse) for _ in range(num_layers)]),
                                                        attention_mechanism=attention_mechanism,
                                                        attention_layer_size=size_layer)
    return attention_out


def random_choise(comment_list):
    """
    # 将数据顺序随机打乱
    :param comment_list: 评论列表
    :param label_list: 标签列表
    :return: 打乱顺序后的评论列表，打乱顺序后的标签列表
    """
    comment_shuffled = []
    num_list_len = len(comment_list)
    # print(seed)
    seed = random.sample(range(num_list_len), num_list_len)
    # print(seed)
    for i in seed:
        comment_shuffled.append(comment_list[i])
    return comment_shuffled


#  创建计算图
train_graph = tf.Graph()
with train_graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(train_graph)
    inputs = tf.placeholder(tf.int32, [batch_size, None], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, None], name='targets')
    # print(35557,targets.shape)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    target_sequence_length = tf.placeholder(tf.int32, (batch_size,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (batch_size,), name='source_sequence_length')
    dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

    with tf.variable_scope("encoder_embedding", dtype=tf.float32):
        en_embedding = tf.get_variable('embedding', shape=[vocab_size, embedding_size])
        # 利用词频计算新的词嵌入矩阵
        # normWordEmbedding = normalize(embedding, vocab_freqs)
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        # print("embedding:",embedding.shape)

    with tf.variable_scope("decoder_embedding", dtype=tf.float32):
        de_embedding = tf.get_variable('embedding', shape=[vocab_size, embedding_size])
        # 利用词频计算新的词嵌入矩阵
        # normWordEmbedding = normalize(embedding, vocab_freqs)
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        # print("embedding:", embedding.shape)
        # de_embedding = en_embedding
    with tf.variable_scope("encoder"):
        # encoder
        # encoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer, dropout_keep_prob) for _ in range(num_layers)])
        encoder_embedded = tf.nn.embedding_lookup(en_embedding, inputs)
        # encoder_embedded = tf.nn.dropout(encoder_embedded, dropout_keep_prob)

        # bidirectional
        encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer,dropout_keep_prob) for _ in range(num_layers)])
        (
            (encoder_fw_outputs, encoder_bw_outputs),
            (encoder_fw_state, encoder_bw_state)
        ) = tf.nn.bidirectional_dynamic_rnn(  # 动态多层双向lstm_rnn
            cell_fw=encoder_cell,
            cell_bw=encoder_cell_bw,
            inputs=encoder_embedded,
            sequence_length=source_sequence_length,
            dtype=tf.float32,
        )
        encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)

        encoder_final_state = []
        for i in range(num_layers):
            c_fw, h_fw = encoder_fw_state[i]
            c_bw, h_bw = encoder_bw_state[i]
            c = tf.concat((c_fw, c_bw), axis=-1)
            h = tf.concat((h_fw, h_bw), axis=-1)
            # print(23332, type(encoder_final_state))
            encoder_final_state.append(tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h))
            # print(23333, type(encoder_final_state))
        encoder_final_state = tuple(encoder_final_state)

        # encoder_out, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
        #                                                          inputs=encoder_embedded,
        #                                                          sequence_length=source_sequence_length,
        #                                                          dtype=tf.float32)

    # decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))

    with tf.variable_scope("decoder"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], word2id['<GO>']), ending], 1)
        # if bidirectional:
        # encoder_state = encoder_final_state[-num_layers:]
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer*2,dropout_keep_prob) for _ in range(num_layers)])
        # decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=size_layer,
            memory=encoder_outputs,
            memory_sequence_length=source_sequence_length
        )

        def cell_input_fn(inputs, attention):
            attn_projection = tf.layers.Dense(size_layer*2,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=decoder_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=size_layer,
            cell_input_fn=cell_input_fn,
            name='AttentionWrapper'
        )

        # 输出层
        output_dense_layer = tf.layers.Dense(vocab_size)

        # 训练
        # if isTrain == 1:
        decoder_embedded = tf.nn.embedding_lookup(de_embedding, decoder_input)
        # decoder_embedded = tf.nn.dropout(decoder_embedded, dropout_keep_prob)
        decoder_initial_state = attention_cell.zero_state(batch_size=batch_size,
                                                dtype=tf.float32).clone(cell_state=encoder_final_state)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embedded,
            sequence_length=target_sequence_length,
            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attention_cell,
            helper=training_helper,
            initial_state=decoder_initial_state,
            output_layer=output_dense_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length)

        # 预测
        start_tokens = tf.tile([tf.constant(word2id['<GO>'], dtype=tf.int32)], [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(de_embedding,
                                                                     start_tokens,
                                                                     word2id['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell,
                                                             predicting_helper,
                                                             decoder_initial_state,
                                                             output_dense_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            maximum_iterations=max_target_sequence_length)

    # training_logits = training_decoder_output.rnn_output
    training_logits = tf.identity(training_decoder_output.rnn_output, name='training_logits')
    predicting_logits = tf.identity(predicting_decoder_output.rnn_output, name='predicting_logits')
    # the result of the prediction
    prediction = tf.identity(predicting_decoder_output.sample_id, 'prediction_result')

    # the score of the beam search prediction
    # bm_score = tf.identity(bm_decoder_output.beam_search_decoder_output.scores, 'bm_prediction_scores')
    # bm_score = bm_decoder_output.beam_search_decoder_output.scores

    # train loss
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    train_cost = tf.contrib.seq2seq.sequence_loss(
        training_logits,
        targets,
        masks)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # compute gradient
    gradients = optimizer.compute_gradients(train_cost)
    # apply gradient clipping to prevent gradient explosion
    capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
    # update the RNN
    train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)


# create session to run the TensorFlow operations
with tf.Session(graph=train_graph) as sess:
    # define summary file writer
    # define saver, keep max_model_number of most recent models
    saver = tf.train.Saver(max_to_keep=max_model_number)
    if isTrain == 1:
        # run initializer
        sess.run(tf.global_variables_initializer())
        # get global step
        # step = tf.train.global_step(sess, global_step)
        step = 0
        best_loss = 1.3
        loss = 0.0
        # train the model
        for epoch_i in range(1, epochs + 1):
            random_train_source = random_choise(train_source)
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(random_train_source,word2id['<PAD>'])):
                _, loss, = sess.run(
                    [train_op, train_cost],
                    {inputs: sources_batch,
                     targets: targets_batch,
                     learning_rate: lr,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths,
                     dropout_keep_prob: 0.6})
                if step % display_step == 0:
                    print(
                        'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}'.format(
                                                   epoch_i,
                                                   epochs,
                                                   batch_i,
                                                   len(train_source) // batch_size,
                                                   loss,
                                                   ))
                # # save the model every epoch
                if loss < best_loss:
                    # 保存最好结果
                    best_loss = loss
                    saver.save(sess, "./ckpt/model.ckpt", global_step=step)
                    with open("log.txt","w",encoding="utf-8")as f:
                        f.write('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}'.format(
                                                   epoch_i,
                                                   epochs,
                                                   batch_i,
                                                   len(train_source) // batch_size,
                                                   loss,
                                                   )+"\n")
                # saver.save(sess, save_path='./ckpt/model.ckpt', global_step=step)
                step += 1
        # save the model when finished
        saver.save(sess, save_path='./ckpt/model.ckpt', global_step=step)
        print('Model Trained and Saved')
    else:
        # load model from folder
        checkpoint = tf.train.latest_checkpoint('./ckpt')
        saver.restore(sess, checkpoint)
        # use the trained model to perform pronunciation prediction
        if isTrain == 0:
            while True:
                test_input = input(">>")
                converted_input = [word2id[c] for c in test_input]
                result = sess.run(
                    [prediction,predicting_logits],
                    {inputs: [converted_input] * batch_size,
                     target_sequence_length: [len(converted_input)] * batch_size,
                     source_sequence_length: [len(converted_input)] * batch_size,
                     dropout_keep_prob: 1
                     })

                test_y = np.argmax(result[-1], axis=2)[1, :]
                # print(test_y)
                tmp = []
                for idx in test_y:
                    if idx == word2id['<EOS>']:
                        break
                    tmp.append(idx2word[idx])
                print("".join(tmp))
