import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
import tensorflow as tf
import random

from Model.SeqGan.Components.Gen_Data_loader import Gen_Data_loader, Dis_dataloader
from Model.SeqGan.Components.Generator import Generator
from Model.SeqGan.Components.Discriminator import Discriminator
from Model.SeqGan.Components.Rollout import Rollout
from Model.SeqGan.Components.TargetLstm import TargetLstm
import cPickle

from Base import Gan


class SeqGan(Gan):

    #########################################################################################
    #  Generator  Hyper-parameters
    ######################################################################################
    EMB_DIM = 32  # embedding dimension
    HIDDEN_DIM = 32  # hidden state dimension of lstm cell
    SEQ_LENGTH = 20  # sequence length
    START_TOKEN = 0
    PRE_EPOCH_NUM = 120  # supervise (maximum likelihood estimation) epochs
    SEED = 88
    BATCH_SIZE = 64

    #########################################################################################
    #  Discriminator  Hyper-parameters
    #########################################################################################
    dis_embedding_dim = 64
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    dis_dropout_keep_prob = 0.75
    dis_l2_reg_lambda = 0.2
    dis_batch_size = 64

    #########################################################################################
    #  Basic Training Parameters
    #########################################################################################
    TOTAL_BATCH = 200
    positive_file = 'save/real_data.txt'
    negative_file = 'save/generator_sample.txt'
    eval_file = 'save/eval_file.txt'
    generated_num = 10000

    def __init__(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        self.gen_data_loader = Gen_Data_loader(self.BATCH_SIZE)
        self.likelihood_data_loader = Gen_Data_loader(self.BATCH_SIZE)  # For testing
        vocab_size = 5000
        self.dis_data_loader = Dis_dataloader(self.BATCH_SIZE)

        self.generator = Generator(self.vocab_size, self.BATCH_SIZE, self.EMB_DIM, self.HIDDEN_DIM, self.SEQ_LENGTH, self.START_TOKEN)
        target_params = cPickle.load(open('save/target_params.pkl'))
        self.target_lstm = TargetLstm(vocab_size, self.BATCH_SIZE, self.EMB_DIM, self.HIDDEN_DIM, self.SEQ_LENGTH, self.START_TOKEN,
                                  target_params)  # The oracle model

        self.discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size,
                                      embedding_size=self.dis_embedding_dim,
                                      filter_sizes=self.dis_filter_sizes, num_filters=self.dis_num_filters,
                                      l2_reg_lambda=self.dis_l2_reg_lambda)

    def _pre_train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        self.__generate_samples(self)
        self.gen_data_loader.create_batches(self.positive_file)

        log = open('save/experiment-log.txt', 'w')
        #  pre-train generator

        print('Start pre-training...')
        log.write('pre-training...\n')

        for epoch in range(self.PRE_EPOCH_NUM):
            loss = self.__pre_train_epoch(self)
            if epoch % 5 == 0:
                self.__generate_samples()
                self.likelihood_data_loader.create_batches(self.eval_file)
                test_loss = self.target_loss(sess, self.target_lstm, self.likelihood_data_loader)
                print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                log.write(buffer)

        'Start pre-training discriminator...'
        # Train 3 epoch on the generated data and do this for 50 times
        for _ in range(50):
            self.__generate_samples(self)
            self.dis_data_loader.load_train_data(self.positive_file, self.negative_file)
            for _ in range(3):
                self.dis_data_loader.reset_pointer()
                for it in range(self.dis_data_loader.num_batch):
                    x_batch, y_batch = self.dis_data_loader.next_batch()
                    feed = {
                        self.discriminator.input_x: x_batch,
                        self.discriminator.input_y: y_batch,
                        self.discriminator.dropout_keep_prob: self.dis_dropout_keep_prob
                    }
                    _ = sess.run(self.discriminator.train_op, feed)


    def adTrain(self):

        rollout = Rollout(self.generator, 0.8)
        print('#########################################################################')
        print('Start Adversarial Training...')
        log.write('adversarial training...\n')
        for total_batch in range(TOTAL_BATCH):
            # Train the generator for one step
            for it in range(1):
                samples = generator.generate(sess)
                rewards = rollout.get_reward(sess, samples, 16, discriminator)
                feed = {generator.x: samples, generator.rewards: rewards}
                _ = sess.run(generator.g_updates, feed_dict=feed)

            # Test
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                print
                'total_batch: ', total_batch, 'test_loss: ', test_loss
                log.write(buffer)

            # Update roll-out parameters
            rollout.update_params()

            # Train the discriminator
            for _ in range(5):
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                dis_data_loader.load_train_data(positive_file, negative_file)

                for _ in range(3):
                    dis_data_loader.reset_pointer()
                    for it in xrange(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: dis_dropout_keep_prob
                        }
                        _ = sess.run(discriminator.train_op, feed)

        log.close()

    def train(self):
        #todo 没写呢
        pass

    def __generate_samples(self):
        # Generate Samples
        generated_samples = []
        for _ in range(int(self.generated_num / self.batch_size)):
            generated_samples.extend(self.trainable_model.generate(self.sess))

        with open(self.output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)

    def __pre_train_epoch(self):
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        self.data_loader.reset_pointer()

        for it in xrange(self.data_loader.num_batch):
            batch = self.data_loader.next_batch()
            _, g_loss = self.trainable_model.pretrain_step(self.sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)



















