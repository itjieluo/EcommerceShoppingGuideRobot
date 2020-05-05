import tensorflow as tf
import numpy as np
import pickle
import data_unit as du
import random

# Learning rate
lr = 0.0001
# Optimizer used by the model, 0 for SGD, 1 for Adam, 2 for RMSProp
optimizer_type = 1
# Mini-batch size
batch_size = 256
# Cell type, 0 for LSTM, 1 for GRU
Cell_type = 0
# Activation function used by RNN cell, 0 for tanh, 1 for relu, 2 for sigmoid
activation_type = 0
# Number of cells in each layer
size_layer = 128
# Number of layers
num_layers = 4
# Embedding size for encoding part and decoding part
embedding_size = 300

# Decoder type, 0 for basic, 1 for beam search
Decoder_type = 1
# Beam width for beam search decoder
beam_width = 2
# Number of max epochs for training
epochs = 30
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = 0
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 5


# 读取数据
# 读取数据
train_data, test_data, word2id, idx2word = du.read_train_data("question_answer.pk")
# print(len(train_data), train_data[:3])
# print(word2id["，"])
# print(idx2word[2498])

# 数据转化为id表示
train_data_int = du.map_int(train_data, word2id)
test_data_int = du.map_int(test_data, word2id)
# print(train_data_int[:3])
# 补齐数据为bath_size整数倍
train_remainder = len(train_data_int) % batch_size
test_remainder = len(train_data_int) % batch_size

# 训练集
train_source = train_data_int + train_data_int[0:batch_size-train_remainder]
# 测试集
test_source = test_data_int + test_data_int[0:batch_size-test_remainder]

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


# 定义生成器，用来获取batch
def get_batches(data, pad_int):
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
#
# for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
#                     get_batches(train_source,word2id['<PAD>'])):
#     print(len(targets_batch),len(targets_batch))

def lstm_cell(dropout_keep_prob, reuse=False):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(size_layer, reuse)
    return tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)


def attention(encoder_out, seq_len, keep_prob, reuse=False):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=size_layer,
                                                               memory=encoder_out,
                                                               memory_sequence_length=seq_len)
    attention_out = tf.contrib.seq2seq.AttentionWrapper(cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell(keep_prob,reuse) for _ in range(num_layers)]),
                                                        attention_mechanism=attention_mechanism,
                                                        attention_layer_size=size_layer)
    return attention_out


def random_choise(comment_list,num):
    """
    # 将数据顺序随机打乱
    :param comment_list: 评论列表
    :param label_list: 标签列表
    :return: 打乱顺序后的评论列表，打乱顺序后的标签列表
    """
    comment_shuffled = []
    num_list_len = len(comment_list)
    # print(seed)
    seed = random.sample(range(num_list_len), num)
    # print(seed)
    for i in seed:
        comment_shuffled.append(comment_list[i])
    return comment_shuffled


# 创建计算图
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

    with tf.variable_scope("embedding", dtype=tf.float32):
        embedding = tf.get_variable('embedding', shape=[vocab_size, embedding_size])
        # 利用词频计算新的词嵌入矩阵
        # normWordEmbedding = normalize(embedding, vocab_freqs)
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        print("embedding:",embedding.shape)

    with tf.variable_scope("encoder"):
        # encoder
        # encoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(embedding, inputs)
        encoder_embedded = tf.nn.dropout(encoder_embedded, dropout_keep_prob)
        encoder_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(dropout_keep_prob) for _ in range(num_layers)])
        encoder_out, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cells,
                                                                 inputs=encoder_embedded,
                                                                 sequence_length=source_sequence_length,
                                                                 dtype=tf.float32)

    ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], word2id['<GO>']), ending], 1)
    # decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    with tf.variable_scope("decoder"):
        # decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
        decoder_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
        decoder_embedded = tf.nn.dropout(decoder_embedded, dropout_keep_prob)
        decoder_cell = attention(encoder_out, source_sequence_length,dropout_keep_prob)
        dense_layer = tf.layers.Dense(vocab_size)

        # training
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embedded,
            sequence_length=target_sequence_length,
            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=training_helper,
            initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
            output_layer=dense_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length)

        # Greedy predict
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile([tf.constant(word2id['<GO>'], dtype=tf.int32)], [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=word2id['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=predicting_helper,
            initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
            output_layer=dense_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length)

        # BeamSearch
        # # dynamic_decode返回(final_outputs, final_state, final_sequence_lengths)。其中：final_outputs是tf.contrib.seq2seq.BasicDecoderOutput类型，包括两个字段：rnn_output，sample_id
        # tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
        # bm_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embeddings, start_tokens,
        #                                                   word2id['<EOS>'], tiled_encoder_state,
        #                                                   beam_width, dense_layer)
        #
        # # impute_finished must be set to false when using beam search decoder
        # # https://github.com/tensorflow/tensorflow/issues/11598
        # bm_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(bm_decoder,
        #                                                             maximum_iterations=2*max_target_sequence_length)

        #  tf.identity 相当与 copy
    # training_logits = tf.identity(training_decoder_output.rnn_output, name='training_logits')
    training_logits = training_decoder_output.rnn_output
    # predicting_logits = tf.identity(predicting_decoder_output.rnn_output, name='predicting_logits')
    predicting_logits = predicting_decoder_output.rnn_output
    # the result of the prediction
    # prediction = tf.identity(predicting_decoder_output.sample_id, 'prediction_result')
    prediction = predicting_decoder_output.sample_id
    # bm_prediction = tf.identity(bm_decoder_output.predicted_ids, 'bm_prediction_result')
    # bm_prediction = bm_decoder_output.predicted_ids
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    # the score of the beam search prediction
    # bm_score = tf.identity(bm_decoder_output.beam_search_decoder_output.scores, 'bm_prediction_scores')
    # bm_score = bm_decoder_output.beam_search_decoder_output.scores

    # train loss
    train_cost = tf.contrib.seq2seq.sequence_loss(
        training_logits,
        targets,
        masks)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # compute gradient
    gradients = optimizer.compute_gradients(train_cost)
    # apply gradient clipping to prevent gradient explosion
    capped_gradients = [(tf.clip_by_norm(grad, 3.), var) for grad, var in gradients if grad is not None]
    # update the RNN
    train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)

    # train_y_t = tf.argmax(training_logits, axis=2)
    # train_y_t = tf.cast(train_y_t, tf.int32)
    # train_prediction = tf.boolean_mask(train_y_t, masks)
    # mask_label = tf.boolean_mask(targets, masks)
    # correct_pred = tf.equal(train_prediction, mask_label)
    # train_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope("validation"):
        # print(233, predicting_logits.eval)
        # get the max length of the predicting result
        val_seq_len = tf.shape(predicting_logits)[1]  #[batchsize,序列长度,embeding]
        # process the predicting result so that it has the same shape with targets
        predicting_logits = tf.concat([predicting_logits, tf.fill(
            [batch_size, max_target_sequence_length - val_seq_len, tf.shape(predicting_logits)[2]], 0.0)], axis=1)
        # print(344, predicting_logits.eval)
        # calculate loss
        validation_cost = tf.contrib.seq2seq.sequence_loss(
            predicting_logits,
            targets,
            masks)

        # test_y_t = tf.argmax(predicting_logits, axis=2)
        # test_y_t = tf.cast(test_y_t, tf.int32)
        # test_prediction = tf.boolean_mask(test_y_t, masks)
        # test_mask_label = tf.boolean_mask(targets, masks)
        # test_correct_pred = tf.equal(test_prediction, test_mask_label)
        # test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))

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
        # train the model
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_source,word2id['<PAD>'])):
                # print(233,targets_lengths)
                _, loss, = sess.run(
                    [train_op, train_cost],
                    {inputs: sources_batch,
                     targets: targets_batch,
                     learning_rate: lr,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths,
                     dropout_keep_prob: 0.2})
                if step % display_step == 0:
                    # calculate the word error rate (WER) and validation loss of the model
                    error = 0.0
                    vali_loss = []
                    val_data = random_choise(test_source, 5000)
                    for (valid_targets_batch, valid_sources_batch, valid_targets_lengths,
                            valid_source_lengths) in get_batches(
                            val_data,
                            word2id['<PAD>']):
                        validation_loss, basic_prediction = sess.run(
                            [validation_cost, prediction],
                            {inputs: valid_sources_batch,
                             targets: valid_targets_batch,
                             learning_rate: lr,
                             target_sequence_length: valid_targets_lengths,
                             source_sequence_length: valid_source_lengths,
                             dropout_keep_prob: 1})

                        vali_loss.append(validation_loss)
                        # error += cal_error(valid_sources_batch, basic_prediction)
                        # error += validation_loss
                    vali_loss = np.mean(vali_loss)
                    # WER = error / len(valid_target)
                    print(
                        'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}'
                        '- Validation loss: {:>6.3f}'.format(epoch_i,
                                                   epochs,
                                                   batch_i,
                                                   len(train_source) // batch_size,
                                                   loss,
                                                   vali_loss,
                                                   # WER,
                                                   ))
                    # save the model every epoch
                    saver.save(sess, save_path='./ckpt/model.ckpt', global_step=step)
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
                # if the decoder type is 0, use the basic decoder, same as set beam width to 0
                if Decoder_type == 0:
                    beam_width = 1
                result = sess.run(
                    [prediction,predicting_logits],
                    {inputs: [converted_input] * batch_size,
                     target_sequence_length: [len(converted_input)] * batch_size,
                     source_sequence_length: [len(converted_input)] * batch_size,
                     dropout_keep_prob: 1
                     })
                # print(24444, result[-1])
                # print(444,result[0])
                # print(442,len(result[0]))
                # print(2333,result[0][1, :, 0])
                # print(555,result[0][1, :, 1])
                # print(666,result[0][1, :, 2])
                # print(777,result[0][2, :, 0])
                # [53 58 49 27 23]
                # print(222,result[0][1,:])
                test_y = np.argmax(result[-1], axis=2)[1,:]
                # print(test_y)
                # tmp = []
                # for idx in test_y:
                #     tmp.append(idx2word[idx])
                # print("".join(tmp))
                tmp = []
                flag = 0
                for idx in test_y:
                    tmp.append(idx2word[idx])
                    if idx == word2id['<EOS>']:
                        print(''.join(tmp))
                        flag = 1
                        break
                # prediction length exceeds the max length
                if not flag:
                    print(' '.join(tmp))

                # print("result:")
                # for i in range(beam_width):
                #     tmp = []
                #     flag = 0
                #     for idx in result[0][1, :, i]:
                #         tmp.append(idx2word[idx])
                #         if idx == word2id['<EOS>']:
                #             print(' '.join(tmp))
                #             flag = 1
                #             break
                #     # prediction length exceeds the max length
                #     if not flag:
                #         print(' '.join(tmp))
                #
                #     # print the score of the result
                #     print('score: {0:.4f}'.format(result[1][0, :, i][-1]))
                #     print('')
        # evaluate the model's performance
        else:
            error = 0.0
            test_loss = []
            for _, (
                    test_targets_batch, test_sources_batch, test_targets_lengths,
                    test_source_lengths) in enumerate(
                get_batches(test_source,
                            word2id['<PAD>'],
                            )
            ):
                validation_loss, basic_prediction = sess.run(
                    [validation_cost, prediction],
                    {inputs: test_sources_batch,
                     targets: test_targets_batch,
                     learning_rate: lr,
                     target_sequence_length: test_targets_lengths,
                     source_sequence_length: test_source_lengths,
                     dropout_keep_prob: 1})

                test_loss.append(validation_loss)
            # calculate the average validation cost and the WER over the validation data set
            test_loss = sum(test_loss) / len(test_loss)
            WER = error / len(test_source)
            print('Test loss: {:>6.3f}'
                  ' - WER: {:>6.2%} '.format(test_loss, WER))





