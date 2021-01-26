from functools import partial
from generative_models import utils
import tensorflow as tf

import numpy as np

tf.compat.v1.disable_eager_execution()

"""
1. Implement Wasserstein loss
"""
class Wasserstein(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.math.reduce_mean(y_true * y_pred, axis=-1)

"""
2. Implement conv block

Conv1D
Activation(LeakyReLU)
MaxPooling1D
"""

def ConvBlock(x):
  c = tf.keras.layers.Conv1D(32, 3, padding="same")(x)
  a = tf.keras.layers.LeakyReLU(alpha=0.2)(c)
  m = tf.keras.layers.MaxPooling1D(pool_size=2)(a)
  return m

"""
3. Implement deconv block

Conv1D
(Batchnorm)
Activation(LeakyReLU)
UpSampling1D
"""

def DeConvBlock(x):
  c = tf.keras.layers.Conv1D(32, 3, padding="same")(x)
  b = tf.keras.layers.BatchNormalization()(c)
  a = tf.keras.layers.LeakyReLU(alpha=0.2)(b)
  u = tf.keras.layers.UpSampling1D(size=2)(a)
  return u

def build_generator(latent_dim, timesteps, batch_size=64, num_classes=100000):
    
    gen_input = tf.keras.layers.Input((latent_dim,))
    label_input = tf.keras.layers.Input((1, ))
    print("labels shape:", label_input.shape)
    label_embed = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(num_classes, latent_dim)(label_input))
    mixed_input = gen_input * label_embed
    print("mixed_input shape:", mixed_input.shape)
    #mixed_input = mixed_input * tf.constant([[1.0, 2.0], [3.0, 4.0]])
    #mixed_input = tf.reshape(mixed_input, (-1, latent_dim))
    #print("mixed_input shape2:", mixed_input.shape)
    #mixed_input = gen_input
    gdense0 = tf.keras.layers.Dense(15)(mixed_input)
    bnorm0 = tf.keras.layers.BatchNormalization()(gdense0)
    gactivation0 = tf.keras.layers.LeakyReLU(alpha=0.2)(bnorm0)
    
    #expand dims before entry into deconv block- something the original paper failed to mention
    gactivation0 = tf.expand_dims(gactivation0, axis=2)
    #print("gactivation shape (after expand):", gactivation0.shape)

    deconv0 = DeConvBlock(gactivation0)
    deconv1 = DeConvBlock(deconv0)
    deconv2 = DeConvBlock(deconv1)
    gconv0 = tf.keras.layers.Conv1D(1, 3, padding="same")(deconv2)
    bnorm1 = tf.keras.layers.BatchNormalization()(gconv0)
    gactivation1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bnorm1)

    #squeeze previously expanded dim now that it's not needed anymore
    gactivation1 = tf.squeeze(gactivation1, axis=-1)

    gdense1 = tf.keras.layers.Dense(timesteps)(gactivation1)
    activation2 = tf.keras.layers.Activation("tanh")(gdense1)

    generator = tf.keras.Model([gen_input, label_input], activation2, name='generator')
    return generator


def build_critic(timesteps, use_mbd, use_packing, packing_degree, num_classes=100000):
    if use_packing:
        input_temp = tf.keras.layers.Input((timesteps, packing_degree + 1))
        critic_input = input_temp
    else:
        input_temp = tf.keras.layers.Input((timesteps,))
        critic_input = tf.expand_dims(input_temp, axis=-1)

    label_input = tf.keras.layers.Input((1,))
    label_embed = tf.expand_dims(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(num_classes, timesteps)(label_input)), axis=-1)
    
    mixed_input = critic_input * label_embed

    conv0 = ConvBlock(mixed_input)
    conv1 = ConvBlock(conv0)
    conv2 = ConvBlock(conv1)
    cactivation0 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    flatten0 = tf.keras.layers.Flatten()(cactivation0)
    
    if use_mbd:
        flatten0 = utils.MinibatchDiscrimination(15, 3)(flatten0)
    
    cdense0 = tf.keras.layers.Dense(50)(flatten0)
    cactivation1 = tf.keras.layers.LeakyReLU(alpha=0.2)(cdense0)
    cdense1 = tf.keras.layers.Dense(15)(cactivation1)
    cactivation2 = tf.keras.layers.LeakyReLU(alpha=0.2)(cdense1)
    cdense2 = tf.keras.layers.Dense(1)(cactivation2)

    critic = tf.keras.Model([input_temp, label_input], cdense2, name='critic')

    return critic


def build_generator_model(generator, critic, latent_dim, timesteps, use_packing, packing_degree, batch_size,
                          generator_lr):
    generator.trainable = True
    critic.trainable = False

    noise_samples = tf.keras.layers.Input((latent_dim,))
    labels = tf.keras.layers.Input((1, ))
    generated_samples = generator([noise_samples, labels])

    if use_packing:
        generated_samples = tf.reshape(generated_samples, (batch_size, timesteps, 1))
        supporting_noise_samples = tf.keras.layers.Input((latent_dim, packing_degree))
        labels2 = tf.keras.layers.Input((1, ))
        
        reshaped_supporting_noise_samples = tf.reshape(supporting_noise_samples, (batch_size * packing_degree, latent_dim))

        supporting_generated_samples = generator([reshaped_supporting_noise_samples, labels2])
        
        supporting_generated_samples = tf.reshape(supporting_generated_samples, (batch_size, timesteps, packing_degree))
        
        merged_generated_samples = tf.keras.layers.Concatenate(axis=-1)([generated_samples, supporting_generated_samples])

        merged_labels = tf.keras.layers.Concatenate()([labels, labels2])

        generated_criticized = critic([merged_generated_samples, merged_labels])

        generator_model = tf.keras.Model([noise_samples, labels, supporting_noise_samples, labels2], generated_criticized, name='generator_model')
        generator_model.compile(optimizer=tf.keras.optimizers.Adam(generator_lr, beta_1=0, beta_2=0.9), loss=utils.wasserstein_loss)
    else:
        generated_criticized = critic([generated_samples, labels])

        generator_model = tf.keras.Model([noise_samples, labels], generated_criticized, name='generator_model')
        generator_model.compile(optimizer=tf.keras.optimizers.Adam(generator_lr, beta_1=0, beta_2=0.9), loss=utils.wasserstein_loss)
    generator_model.summary()
    return generator_model


def build_critic_model(generator, critic, latent_dim, timesteps, use_packing, packing_degree, batch_size, critic_lr,
                       gradient_penality_weight):
    generator.trainable = False
    critic.trainable = True

    noise_samples = tf.keras.layers.Input((latent_dim,))
    real_samples = tf.keras.layers.Input((timesteps,))
    
    labels = tf.keras.layers.Input((1, ))

    if use_packing:
        supporting_noise_samples = tf.keras.layers.Input((latent_dim, packing_degree))
        supporting_real_samples = tf.keras.layers.Input((timesteps, packing_degree))
        labels2 = tf.keras.layers.Input((1, ))

        reshaped_supporting_noise_samples = tf.keras.layers.Reshape((batch_size * packing_degree, latent_dim))(supporting_noise_samples)
        
        generated_samples = generator([noise_samples, labels])
        supporting_generated_samples = generator([reshaped_supporting_noise_samples, labels2])
        
        expanded_generated_samples = tf.keras.layers.Reshape((batch_size, timesteps, 1))(generated_samples)
        expanded_generated_supporting_samples = tf.keras.layers.Reshape((batch_size, timesteps, packing_degree))(supporting_generated_samples)

        merged_generated_samples = tf.keras.layers.Concatenate(axis=-1)([expanded_generated_samples, expanded_generated_supporting_samples])

        merged_labels = tf.keras.layers.Concatenate()([labels, labels2])

        generated_criticized = critic([merged_generated_samples, merged_labels])
        
        expanded_real_samples = tf.keras.layers.Reshape((batch_size, timesteps, 1))(real_samples)
        merged_real_samples = tf.keras.layers.Concatenate(axis=-1)([expanded_real_samples, supporting_real_samples])
        
        real_criticized = critic([merged_real_samples, merged_labels])
        print("RWA1 inputs shapes:", real_samples.shape, generated_samples.shape)
        averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
        
        expanded_averaged_samples = tf.reshape(averaged_samples, (batch_size, timesteps, 1))
        
        expanded_supporting_real_samples = tf.reshape(supporting_real_samples, (batch_size * packing_degree, timesteps))
        print("RWA2 inputs shapes:", expanded_supporting_real_samples.shape, supporting_generated_samples.shape)
        averaged_support_samples = RandomWeightedAverage((batch_size * packing_degree))(
            [expanded_supporting_real_samples, supporting_generated_samples])
        
        averaged_support_samples = tf.reshape(averaged_support_samples, (batch_size, timesteps, packing_degree))

        merged_averaged_samples = tf.keras.layers.Concatenate(axis=-1)([expanded_averaged_samples, averaged_support_samples])
        averaged_criticized = critic([merged_averaged_samples, merged_labels])
        
        """
        with tf.GradientTape() as t:
          t.watch(merged_averaged_samples)
          averaged_criticized = critic(merged_averaged_samples)
          
        """

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=merged_averaged_samples,
                                  gradient_penalty_weight=gradient_penality_weight)

        """
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=merged_averaged_samples,
                                  gradient_penalty_weight=gradient_penality_weight,
                                  gradient_tape = t)
        """
        partial_gp_loss.__name__ = 'gradient_penalty'

        critic_model = tf.keras.Model([real_samples, noise_samples, labels, supporting_real_samples, supporting_noise_samples, labels2],
                             [real_criticized, generated_criticized, averaged_criticized], name='critic_model')

        critic_model.compile(optimizer=tf.keras.optimizers.Adam(critic_lr, beta_1=0, beta_2=0.9),
                             loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss],
                             loss_weights=[1 / 3, 1 / 3, 1 / 3])
    else:
        generated_samples = generator([noise_samples, labels])
        generated_criticized = critic([generated_samples, labels])
        real_criticized = critic([real_samples, labels])

        print("RWA3 inputs shapes:", real_samples.shape, generated_samples.shape)
        averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
        averaged_criticized = critic([averaged_samples, labels])
        """
        with tf.GradientTape() as t:
          t.watch(averaged_samples)
          averaged_criticized = critic(averaged_samples)
        """
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=gradient_penality_weight)

        """
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=gradient_penality_weight,
                                  gradient_tape = t)
        """
        partial_gp_loss.__name__ = 'gradient_penalty'

        critic_model = tf.keras.Model([real_samples, noise_samples, labels],
                             [real_criticized, generated_criticized, averaged_criticized], name='critic_model')

        critic_model.compile(optimizer=tf.keras.optimizers.Adam(critic_lr, beta_1=0, beta_2=0.9),
                             loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss],
                             loss_weights=[1 / 3, 1 / 3, 1 / 3])
    critic_model.summary()
    return critic_model


def gradient_penalty_loss(_, y_pred, averaged_samples, gradient_penalty_weight, gradient_tape=None):
    #print("y_pred:", y_pred)
    #print("averaged_samples:", averaged_samples)
    gradients = tf.gradients(y_pred, averaged_samples)
    #gradients = gradient_tape.gradient(y_pred, [averaged_samples])
    #print("gradients:", gradients)
    gradients = gradients[0]
    gradients_sqr = tf.math.square(gradients)
    gradients_sqr_sum = tf.math.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = tf.math.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * tf.math.square(1 - gradient_l2_norm)
    return tf.math.reduce_mean(gradient_penalty)


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size
        
        #self.w = tf.random.uniform((self._batch_size, 1))
        self.w = self.add_weight("w", shape=(self._batch_size, 1), trainable=True, initializer=tf.keras.initializers.RandomUniform())

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        averaged_inputs = (self.w * inputs[0]) + ((1 - self.w) * inputs[1])
        return averaged_inputs
      
    def compute_output_shape(self, input_shape):
        return input_shape[0]
     
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            '_batch_size': self._batch_size
        })
        return config
