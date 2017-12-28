import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import ceil

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)


class VariationalAutoEncoder():
	def __init__(self,latent_dim):
		self.latent_dim = latent_dim
		self.Build()
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

	def Encoder(self):
		temp1 = tf.layers.dense(self.x,512,tf.nn.elu)
		temp2 = tf.layers.dense(temp1,384,tf.nn.elu)
		temp3 = tf.layers.dense(temp2,256,tf.nn.elu)
		mean = tf.layers.dense(temp3,self.latent_dim)
		log_std_sqd = tf.layers.dense(temp3,self.latent_dim)
		return (mean,log_std_sqd)

	def Decoder(self):
		temp4 = tf.layers.dense(self.z_sampled,256,tf.nn.elu)
		temp5 = tf.layers.dense(temp4,384,tf.nn.elu)
		temp6 = tf.layers.dense(temp5,512,tf.nn.elu)
		decoded = tf.layers.dense(temp6,784,tf.sigmoid)
		return decoded

	def Build(self):
		self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 784])
		self.mean,self.log_std_sqd = self.Encoder()
		std = tf.sqrt(tf.exp(self.log_std_sqd))
		self.z_sampled = tf.random_normal(tf.shape(self.log_std_sqd))
		self.z_sampled = tf.add(tf.multiply(self.z_sampled,std),self.mean)	
		self.decoded = self.Decoder()
		
		generation_loss = -tf.reduce_sum(self.x * tf.log(1e-8 + self.decoded) + (1-self.x) * tf.log(1e-8 + 1 - self.decoded),1)
		generation_loss = tf.reduce_mean(generation_loss)
		

		latent_loss = 0.5*tf.reduce_sum(tf.square(self.mean) + tf.exp(self.log_std_sqd) - self.log_std_sqd + 1)
		latent_loss = tf.reduce_mean(latent_loss)
		

		loss = tf.add(generation_loss,latent_loss)

		self.latent_loss,self.generation_loss,self.loss = latent_loss,generation_loss,loss

		
		optimizer = tf.train.RMSPropOptimizer(3e-4)
		self.update = optimizer.minimize(self.loss)
		return

	def Run_a_Batch(self,batch):
		total_loss,generation_loss,latent_loss,_ = self.sess.run(
			[self.loss,self.generation_loss,self.latent_loss,self.update],feed_dict = {self.x:batch}
			)	
		return (total_loss,generation_loss,latent_loss)
	
	def reconstructor(self, x):
		x_hat = self.sess.run(self.decoded, feed_dict={self.x: x})
		return x_hat

	# z -> x
	def generator(self, z):
		x_hat = self.sess.run(self.decoded, feed_dict={self.z_sampled: z})
		return x_hat

	# x -> z
	def transformer(self, x):
		z = self.sess.run(self.z_sampled, feed_dict={self.x: x})
		return z



epochs = 1
batch_size = 50
num_sample = mnist.train.num_examples
num_runs_in_epoch = 1000#ceil(num_sample/batch_size) + 5

model = VariationalAutoEncoder(10)

for _ in range(epochs):
	for _ in range(num_runs_in_epoch):
		batch = mnist.train.next_batch(batch_size)
		total_loss,generation_loss,latent_loss = model.Run_a_Batch(batch[0])
		print("%f + %f = %f"%(generation_loss,latent_loss,total_loss))


batch = mnist.test.next_batch(5)
x_reconstructed = model.reconstructor(batch[0])



batch_reshaped = np.reshape(batch[0],(5,28,28))
x_reconstructed_reshaped = np.reshape(x_reconstructed,(5,28,28))

#-------------------------------------x-----------------x----------------------------------#
fig = plt.figure()
for temp_var in range(5):
	plt.subplot(5, 2, 2*temp_var+1)
	plt.imshow(batch_reshaped[temp_var])
	plt.subplot(5, 2, 2*temp_var+2)
	plt.imshow(x_reconstructed_reshaped[temp_var])	
plt.savefig('Reconstructed_Images')
plt.close(fig)
#-------------------------------------x-----------------x----------------------------------#


z = np.random.normal(size=[25, model.latent_dim])
x_generated = model.generator(z)

x_generated_reshaped = np.reshape(x_generated,(25,28,28))
#-------------------------------------x-----------------x----------------------------------#
fig = plt.figure()
for temp_var in range(25):
	plt.subplot(5,5,temp_var+1)
	plt.imshow(x_generated_reshaped[temp_var])
plt.savefig('Generated_Images')
plt.close(fig)	
#-------------------------------------x-----------------x----------------------------------#
