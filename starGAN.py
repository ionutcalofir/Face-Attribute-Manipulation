import os
import cv2
import pickle
import shutil
import numpy as np
import tensorflow as tf

from utils import Utils

class StarGAN:
  def __init__(self,
               img_height=64,
               img_width=64,
               img_c=3,
               nd=7,
               nc=7,
               batch_size=8,
               no_epochs=2,
               lambda_cls=1,
               lambda_rec=10,
               learning_rate=0.0001,
               adam_beta1=0.5,
               adam_beta2=0.999):
    """
    img_height - height of image
    img_width - width of image
    img_c - number of channels of image
    nd - number of domains
    nc - dimension of domain labels
    batch_size - size of the batch
    no_epochs - number of epochs
    lambda_cls - controls the relative importance of domain classification
    lambda_rec - controls the relative importance of reconstruction
    learning_rate - the learning rate
    adam_beta1 - hyper-parameter for adam optimizer
    adam_beta2 - hyper-parameter for adam optimizer
    """
    self.img_height = img_height
    self.img_width = img_width
    self.img_c = img_c
    self.nd = nd
    self.nc = nc
    self.batch_size = batch_size
    self.no_epochs = no_epochs
    self.lambda_cls = lambda_cls
    self.lambda_rec = lambda_rec
    self.learning_rate = learning_rate
    self.adam_beta1 = adam_beta1
    self.adam_beta2 = adam_beta2

  def _generator(self, img, c, reuse=False, name='generator'):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()

      # labels concatenated to image
      with tf.variable_scope('reshape'):
        img_reshape = tf.concat(
                        [img, tf.tile(c, [1, self.img_height, self.img_width, 1])],
                        3)

      # down-sampling
      with tf.variable_scope('down-sampling'):
        with tf.variable_scope('conv1'):
          W_conv1 = self._weight_variable([7, 7, self.img_c + self.nc, 64])
          b_conv1 = self._bias_variable([64])
          h_conv1 = tf.nn.relu(self._instance_normalization(
                                  self._conv2d(img_reshape, W_conv1,
                                               strides=[1, 1, 1, 1],
                                               padding=[0, 3, 3, 0]))
                               + b_conv1)

          if reuse == False:
            tf.summary.histogram("weight", W_conv1)
            tf.summary.histogram("bias", b_conv1)
            tf.summary.histogram("activation", h_conv1)

        with tf.variable_scope('conv2'):
          W_conv2 = self._weight_variable([4, 4, 64, 128])
          b_conv2 = self._bias_variable([128])
          h_conv2 = tf.nn.relu(self._instance_normalization(
                                  self._conv2d(h_conv1, W_conv2,
                                               strides=[1, 2, 2, 1],
                                               padding=[0, 1, 1, 0]))
                               + b_conv2)

          if reuse == False:
            tf.summary.histogram("weight", W_conv2)
            tf.summary.histogram("bias", b_conv2)
            tf.summary.histogram("activation", h_conv2)

        with tf.variable_scope('conv3'):
          W_conv3 = self._weight_variable([4, 4, 128, 256])
          b_conv3 = self._bias_variable([256])
          h_conv3 = tf.nn.relu(self._instance_normalization(
                                  self._conv2d(h_conv2, W_conv3,
                                               strides=[1, 2, 2, 1],
                                               padding=[0, 1, 1, 0]))
                               + b_conv3)

          if reuse == False:
            tf.summary.histogram("weight", W_conv3)
            tf.summary.histogram("bias", b_conv3)
            tf.summary.histogram("activation", h_conv3)

      # bottleneck
      with tf.variable_scope('bottleneck'):
        with tf.variable_scope('res_block1'):
          W_res_block1 = self._weight_variable([3, 3, 256, 256])
          b_res_block1 = self._bias_variable([256])
          h_res_block1 = tf.nn.relu(self._instance_normalization(
                                      self._residual_block(h_conv3, W_res_block1,
                                                           strides=[1, 1, 1, 1],
                                                           padding=[0, 1, 1, 0]))
                                    + b_res_block1)

          if reuse == False:
            tf.summary.histogram("weight", W_res_block1)
            tf.summary.histogram("bias", b_res_block1)
            tf.summary.histogram("activation", h_res_block1)

        with tf.variable_scope('res_block2'):
          W_res_block2 = self._weight_variable([3, 3, 256, 256])
          b_res_block2 = self._bias_variable([256])
          h_res_block2 = tf.nn.relu(self._instance_normalization(
                                      self._residual_block(h_res_block1, W_res_block2,
                                                           strides=[1, 1, 1, 1],
                                                           padding=[0, 1, 1, 0]))
                                    + b_res_block2)

          if reuse == False:
            tf.summary.histogram("weight", W_res_block2)
            tf.summary.histogram("bias", b_res_block2)
            tf.summary.histogram("activation", h_res_block2)

        with tf.variable_scope('res_block3'):
          W_res_block3 = self._weight_variable([3, 3, 256, 256])
          b_res_block3 = self._bias_variable([256])
          h_res_block3 = tf.nn.relu(self._instance_normalization(
                                      self._residual_block(h_res_block2, W_res_block3,
                                                           strides=[1, 1, 1, 1],
                                                           padding=[0, 1, 1, 0]))
                                    + b_res_block3)

          if reuse == False:
            tf.summary.histogram("weight", W_res_block3)
            tf.summary.histogram("bias", b_res_block3)
            tf.summary.histogram("activation", h_res_block3)

        with tf.variable_scope('res_block4'):
          W_res_block4 = self._weight_variable([3, 3, 256, 256])
          b_res_block4 = self._bias_variable([256])
          h_res_block4 = tf.nn.relu(self._instance_normalization(
                                      self._residual_block(h_res_block3, W_res_block4,
                                                           strides=[1, 1, 1, 1],
                                                           padding=[0, 1, 1, 0]))
                                    + b_res_block4)

          if reuse == False:
            tf.summary.histogram("weight", W_res_block4)
            tf.summary.histogram("bias", b_res_block4)
            tf.summary.histogram("activation", h_res_block4)

        with tf.variable_scope('res_block5'):
          W_res_block5 = self._weight_variable([3, 3, 256, 256])
          b_res_block5 = self._bias_variable([256])
          h_res_block5 = tf.nn.relu(self._instance_normalization(
                                      self._residual_block(h_res_block4, W_res_block5,
                                                          strides=[1, 1, 1, 1],
                                                          padding=[0, 1, 1, 0]))
                                    + b_res_block5)

          if reuse == False:
            tf.summary.histogram("weight", W_res_block5)
            tf.summary.histogram("bias", b_res_block5)
            tf.summary.histogram("activation", h_res_block5)

        with tf.variable_scope('res_block6'):
          W_res_block6 = self._weight_variable([3, 3, 256, 256])
          b_res_block6 = self._bias_variable([256])
          h_res_block6 = tf.nn.relu(self._instance_normalization(
                                      self._residual_block(h_res_block5, W_res_block6,
                                                           strides=[1, 1, 1, 1],
                                                           padding=[0, 1, 1, 0]))
                                    + b_res_block6)

          if reuse == False:
            tf.summary.histogram("weight", W_res_block6)
            tf.summary.histogram("bias", b_res_block6)
            tf.summary.histogram("activation", h_res_block6)

      # up-sampling
      with tf.variable_scope('up-sampling'):
        with tf.variable_scope('deconv1'):
          W_deconv1 = self._weight_variable([4, 4, 128, 256])
          b_deconv1 = self._bias_variable([128])
          output_shape = tf.constant([self.batch_size,
                                      np.asscalar(np.array([self.img_height / 2],
                                                            dtype=np.int32)),
                                      np.asscalar(np.array([self.img_width / 2],
                                                            dtype=np.int32)),
                                      128],
                                      dtype=tf.int32)
          h_deconv1 = tf.nn.relu(self._instance_normalization(
                                    self._deconv2d(h_res_block6, W_deconv1,
                                                   output_shape=output_shape,
                                                   strides=[1, 2, 2, 1],
                                                   padding=[0, 1, 1, 0]))
                                 + b_deconv1)

          if reuse == False:
            tf.summary.histogram("weight", W_deconv1)
            tf.summary.histogram("bias", b_deconv1)
            tf.summary.histogram("activation", h_deconv1)

        with tf.variable_scope('deconv2'):
          W_deconv2 = self._weight_variable([4, 4, 64, 128])
          b_deconv2 = self._bias_variable([64])
          output_shape = tf.constant([self.batch_size,
                                      self.img_height,
                                      self.img_width,
                                      64],
                                      dtype=tf.int32)
          h_deconv2 = tf.nn.relu(self._instance_normalization(
                                    self._deconv2d(h_deconv1, W_deconv2,
                                                   output_shape=output_shape,
                                                   strides=[1, 2, 2, 1],
                                                   padding=[0, 1, 1, 0]))
                                 + b_deconv2)

          if reuse == False:
            tf.summary.histogram("weight", W_deconv2)
            tf.summary.histogram("bias", b_deconv2)
            tf.summary.histogram("activation", h_deconv2)

        with tf.variable_scope('conv1'):
          W_conv1 = self._weight_variable([7, 7, 64, 3])
          b_conv1 = self._bias_variable([3])
          h_out = tf.nn.tanh(self._conv2d(h_deconv2, W_conv1,
                                            strides=[1, 1, 1, 1],
                                            padding=[0, 3, 3, 0])
                             + b_conv1)

          if reuse == False:
            tf.summary.histogram("weight", W_conv1)
            tf.summary.histogram("bias", b_conv1)
            tf.summary.histogram("activation", h_conv1)

          if reuse == False:
            tf.summary.histogram("activations", h_out)

    return h_out

  def _discriminator(self, img, reuse=False, name='discriminator'):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()

      # input layer
      with tf.variable_scope('input-layer'):
        with tf.variable_scope('conv1'):
          W_conv1 = self._weight_variable([4, 4, self.img_c, 64])
          b_conv1 = self._bias_variable([64])
          h_conv1 = tf.nn.leaky_relu(self._conv2d(img, W_conv1,
                                                  strides=[1, 2, 2, 1],
                                                  padding=[0, 1, 1, 0])
                                     + b_conv1)

          if reuse == False:
            tf.summary.histogram("weight", W_conv1)
            tf.summary.histogram("bias", b_conv1)
            tf.summary.histogram("activation", h_conv1)

      # hidden layer
      with tf.variable_scope('hidden-layer'):
        with tf.variable_scope('conv1'):
          W_conv1 = self._weight_variable([4, 4, 64, 128])
          b_conv1 = self._bias_variable([128])
          h_conv1 = tf.nn.leaky_relu(self._conv2d(h_conv1, W_conv1,
                                                  strides=[1, 2, 2, 1],
                                                  padding=[0, 1, 1, 0])
                                     + b_conv1)

          if reuse == False:
            tf.summary.histogram("weight", W_conv1)
            tf.summary.histogram("bias", b_conv1)
            tf.summary.histogram("activation", h_conv1)

        with tf.variable_scope('conv2'):
          W_conv2 = self._weight_variable([4, 4, 128, 256])
          b_conv2 = self._bias_variable([256])
          h_conv2 = tf.nn.leaky_relu(self._conv2d(h_conv1, W_conv2,
                                                  strides=[1, 2, 2, 1],
                                                  padding=[0, 1, 1, 0])
                                     + b_conv2)

          if reuse == False:
            tf.summary.histogram("weight", W_conv2)
            tf.summary.histogram("bias", b_conv2)
            tf.summary.histogram("activation", h_conv2)

        with tf.variable_scope('conv3'):
          W_conv3 = self._weight_variable([4, 4, 256, 512])
          b_conv3 = self._bias_variable([512])
          h_conv3 = tf.nn.leaky_relu(self._conv2d(h_conv2, W_conv3,
                                                  strides=[1, 2, 2, 1],
                                                  padding=[0, 1, 1, 0])
                                     + b_conv3)

          if reuse == False:
            tf.summary.histogram("weight", W_conv3)
            tf.summary.histogram("bias", b_conv3)
            tf.summary.histogram("activation", h_conv3)

        with tf.variable_scope('conv4'):
          W_conv4 = self._weight_variable([4, 4, 512, 1024])
          b_conv4 = self._bias_variable([1024])
          h_conv4 = tf.nn.leaky_relu(self._conv2d(h_conv3, W_conv4,
                                                  strides=[1, 2, 2, 1],
                                                  padding=[0, 1, 1, 0])
                                     + b_conv4)

          if reuse == False:
            tf.summary.histogram("weight", W_conv4)
            tf.summary.histogram("bias", b_conv4)
            tf.summary.histogram("activation", h_conv4)

        with tf.variable_scope('conv5'):
          W_conv5 = self._weight_variable([4, 4, 1024, 2048])
          b_conv5 = self._bias_variable([2048])
          h_conv5 = tf.nn.leaky_relu(self._conv2d(h_conv4, W_conv5,
                                                  strides=[1, 2, 2, 1],
                                                  padding=[0, 1, 1, 0])
                                     + b_conv5)

          if reuse == False:
            tf.summary.histogram("weight", W_conv5)
            tf.summary.histogram("bias", b_conv5)
            tf.summary.histogram("activation", h_conv5)

      with tf.variable_scope('output-layer'):
        with tf.variable_scope('src-conv1'):
          W_src = self._weight_variable([3, 3, 2048, 1])
          b_src = self._bias_variable([1])
          h_src = self._conv2d(h_conv5, W_src,
                               strides=[1, 1, 1, 1],
                               padding=[0, 1, 1, 0]) \
                  + b_src

          if reuse == False:
            tf.summary.histogram("weight", W_src)
            tf.summary.histogram("bias", b_src)

        with tf.variable_scope('cls-conv1'):
          W_cls = self._weight_variable([np.asscalar(np.array([self.img_height / 64],
                                                              dtype=np.int32)),
                                         np.asscalar(np.array([self.img_width / 64],
                                                              dtype=np.int32)),
                                         2048,
                                         self.nd])
          b_cls = self._bias_variable([self.nd])
          h_cls = self._conv2d(h_conv5, W_cls,
                               strides=[1, 1, 1, 1],
                               padding=[0, 0, 0, 0]) \
                  + b_cls

          if reuse == False:
            tf.summary.histogram("weight", W_cls)
            tf.summary.histogram("bias", b_cls)

      if reuse == False:
        tf.summary.histogram("activation_src", h_src)
        tf.summary.histogram("activation_cls", h_cls)

      return h_src, h_cls

  def _instance_normalization(self, x, name='instance_normalization'):
    with tf.variable_scope(name):
      depth = x.get_shape()[3]
      scale = tf.get_variable('scale',
                [depth],
                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
      offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
      mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
      epsilon = 1e-5
      inv = tf.rsqrt(variance + epsilon)
      normalized = (x - mean) * inv

      return scale * normalized + offset

  def _conv2d(self, x, W, strides=[1, 1, 1, 1], padding=[0, 0, 0, 0]):
    """
    returns a 2d convolutional layer
    """
    x_pad = tf.pad(x,
                   [[0, 0], [padding[1], padding[1]],
                    [padding[2], padding[2]], [0, 0]],
                   'CONSTANT')
    return tf.nn.conv2d(x_pad, W, strides, padding='VALID')

  def _deconv2d(self, x, W, output_shape, strides=[1, 1, 1, 1], padding=[0, 0, 0, 0]):
    """
    return a 2d deconvolutional layer
    """
    # you can't add padding to the output matrix with this function,
    # so i set the padding='SAME'
    return tf.nn.conv2d_transpose(x, W, output_shape, strides, padding='SAME')

  def _weight_variable(self, shape):
    """
    generates a variable of a given shape
    """
    return tf.get_variable(name='W', shape=shape, initializer=None)

  def _bias_variable(self, shape):
    """
    generates a bias variable of a given shape
    """
    return tf.get_variable(name='b', shape=shape, initializer=None)

  def _residual_block(self, x, W, strides=[1, 1, 1, 1], padding=[0, 0, 0, 0]):
    """
    returns a residual block
    """
    x_pad = tf.pad(x,
                   [[0, 0], [padding[1], padding[1]],
                    [padding[2], padding[2]], [0, 0]],
                   'CONSTANT', name='pad')
    res = tf.nn.conv2d(x_pad, W, strides, padding='VALID')
    return x + res

  def build(self):
    tf.reset_default_graph()

    self.x = tf.placeholder(tf.float32, [None, self.img_height,
                                         self.img_width, self.img_c])
    self.y = tf.placeholder(tf.float32, [None, 1, 1, self.nc])
    self.y_target = tf.placeholder(tf.float32, [None, 1, 1, self.nc])

    self.img_g = self._generator(self.x, self.y_target)
    self.img_gg = self._generator(self.img_g, self.y, reuse=True)
    self.h_src_real, self.h_cls_real = self._discriminator(self.x)
    self.h_src_fake, self.h_cls_fake = self._discriminator(self.img_g, reuse=True)

    tf.summary.image('original', self.x, 8)
    tf.summary.image('generated', self.img_g, 8)
    tf.summary.image('original_generated', self.img_gg, 8)

    self.src_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.h_src_real,
                            labels=tf.ones_like(self.h_src_real),
                            name='src_real_sigmoid'),
                          name='src_real_loss')
    self.src_fake_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.h_src_fake,
                            labels=tf.zeros_like(self.h_src_fake),
                            name='src_fake_sigmoid'),
                          name='src_fake_loss')

    self.src_fake_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.h_src_fake,
                            labels=tf.ones_like(self.h_src_fake),
                            name='src_fake_sigmoid'),
                          name='src_fake_loss')

    self.cls_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.h_cls_real,
                            labels=self.y,
                            name='cls_real_sigmoid'),
                          name='cls_real_loss')
    self.cls_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.h_cls_fake,
                            labels=self.y_target,
                            name='cls_fake_sigmoid'),
                          name='cls_fake_loss')
    self.rec_loss = tf.reduce_mean(tf.abs(self.x - self.img_gg), name='rec_loss')

    self.d_loss = self.src_real_loss \
                  + self.src_fake_loss_d \
                  + self.lambda_cls * self.cls_real_loss
    self.g_loss = self.src_fake_loss_g \
                  + self.lambda_cls * self.cls_fake_loss \
                  + self.lambda_rec * self.rec_loss

    tf.summary.scalar('src_real_loss', self.src_real_loss)
    tf.summary.scalar('src_fake_loss_d', self.src_fake_loss_d)
    tf.summary.scalar('src_fake_loss_g', self.src_fake_loss_g)
    tf.summary.scalar('cls_real_loss', self.cls_real_loss)
    tf.summary.scalar('cls_fake_loss', self.cls_fake_loss)
    tf.summary.scalar('rec_loss', self.rec_loss)
    tf.summary.scalar('d_loss', self.d_loss)
    tf.summary.scalar('g_loss', self.g_loss)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    self.g_vars = [var for var in t_vars if 'generator' in var.name]

    self.d_optim = tf.train.AdamOptimizer(
                      learning_rate=self.learning_rate,
                      beta1=self.adam_beta1,
                      beta2=self.adam_beta2,
                      name='adam_discriminator').minimize(self.d_loss, var_list=self.d_vars)
    self.g_optim = tf.train.AdamOptimizer(
                      learning_rate=self.learning_rate,
                      beta1=self.adam_beta1,
                      beta2=self.adam_beta2,
                      name='adam_generator').minimize(self.g_loss, var_list=self.g_vars)

  def train(self, resume=False):
    saver = tf.train.Saver()
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
      tensorboard_path = 'tensorboard'
      ep = 0
      it = 0
      summ_count = 0
      train_resume = 0

      if resume == True:
        with open('checkpoints/model_path.pkl', 'rb') as f:
          model_path = pickle.load(f)
        saver.restore(sess, model_path)

        with open('checkpoints/utils.pkl', 'rb') as f:
          ut = pickle.load(f)
        with open('checkpoints/epoch.pkl', 'rb') as f:
          ep = pickle.load(f)
        with open('checkpoints/iteration.pkl', 'rb') as f:
          it = pickle.load(f) + 1
        with open('checkpoints/summ_count.pkl', 'rb') as f:
          summ_count = pickle.load(f) + 1

        with open('checkpoints/train_resume.pkl', 'rb') as f:
          train_resume = pickle.load(f) + 1
        tensorboard_path = tensorboard_path + '/train_resume_' + str(train_resume) \
                           + '_' + str(summ_count)
      else:
        sess.run(tf.global_variables_initializer())

        if os.path.exists(tensorboard_path):
          shutil.rmtree(tensorboard_path)
        os.mkdir(tensorboard_path)
        tensorboard_path = tensorboard_path + '/train'

      os.mkdir(tensorboard_path)
      writer = tf.summary.FileWriter(tensorboard_path)

      if resume == False:
        writer.add_graph(sess.graph)

      while ep < self.no_epochs:
        if resume == False: # if resume=True use ut from the last checkpoint
          ut = Utils() # shuffle data every epoch
        else:
          resume = False

        while True:
          x_batch, y_batch, y_target_batch, batch_end = ut.next_batch_train(self.batch_size)

          if batch_end == True: # end of epoch
            break

          if summ_count % 5 == 0:
            s = sess.run(
                  summ,
                  feed_dict={self.x: x_batch,
                             self.y: y_batch,
                             self.y_target: y_target_batch})
            writer.add_summary(s, summ_count + 10)

          for _ in list(range(5)):
            _, d_loss = sess.run(
                [self.d_optim, self.d_loss],
                feed_dict={self.x: x_batch,
                           self.y: y_batch,
                           self.y_target: y_target_batch})

          _, g_loss = sess.run(
              [self.g_optim, self.g_loss],
              feed_dict={self.x: x_batch,
                         self.y: y_batch,
                         self.y_target: y_target_batch})

          print('epoch: ' + str(ep) + ', iteration: ' + str(it))
          if it % 100 == 0:
            print('epoch: ' + str(ep) + ', iteration: ' + str(it) + ', '
                  + 'g_loss: ' + str(g_loss) + ', '
                  + 'd_loss: ' + str(d_loss))

            model_path = 'models/model_ep' + str(ep) + '_it' + str(it) + '.ckpt'
            saver.save(sess, model_path)

            with open('checkpoints/model_path.pkl', 'wb') as f:
              pickle.dump(model_path, f)
            with open('checkpoints/utils.pkl', 'wb') as f:
              pickle.dump(ut, f)
            with open('checkpoints/epoch.pkl', 'wb') as f:
              pickle.dump(ep, f)
            with open('checkpoints/iteration.pkl', 'wb') as f:
              pickle.dump(it, f)
            with open('checkpoints/summ_count.pkl', 'wb') as f:
              pickle.dump(summ_count, f)
            with open('checkpoints/train_resume.pkl', 'wb') as f:
              pickle.dump(train_resume, f)

            with open('checkpoints/info.txt', 'w') as f:
              f.write('model_path: ' + model_path + '\n')
              f.write('epoch: ' + str(ep) + '\n')
              f.write('iteration: ' + str(it) + '\n')
              f.write('summ_count: ' + str(summ_count) + '\n')
              f.write('train_resume: ' + str(train_resume) + '\n')

          summ_count = summ_count + 1
          it = it + 1

        ep = ep + 1
