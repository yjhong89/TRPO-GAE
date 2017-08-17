import numpy as np
import tensorflow as tf
import time, os
from utils import *
from ops import *


class TRPO():
	def __init__(self, args, env, sess):
		self.args = args
		self.sess = sess
		self.env = env
		
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		print('Observation space', self.observation_space)
		print('Action space', self.action_space)
		#
		self.observation_size = self.env.observation.shape[0]
		# np.prod : return the product of array element over a given axis
		self.action_size = np.prod(self.action_space.shape)

		# Build model and create variables
		self.build_model()

	def build_model(self):
		self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
		self.action = tf.placeholder(tf.float32, [None, self.action_size])
		self.advantage = tf.placeholder(tf.float32, [None])
		# Mean of old action distribution
		self.old_action_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
		self.old_action_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])
		'''
			Mean value for each action : each action has gaussian distribution with mean and standard deviation
			With continuous state and action space, use GAUSSIAN DISTRIBUTION, maps  from the input features to the mean of Gaussian distribution for each action
			Sperate set of parameters specifies the log standard deviation of each action
			=> The policy ius defined by the normnal distribution (mean=NeuralNet(states), stddev= exp(r))
		'''
		self.action_dist_mu, action_dist_logstd = self.build_policy(self.obs)
		# Make log standard shape from [1, action size] => [batch size, action size]
		# tf.tile(A, reps) : construct an tensor by repeating A given by 'reps'
		self.action_dist_logstd = tf.tile(action_dist_logstd, [self.action_dist_mu.get_shape()[0], 1])

		# outputs probability of taking 'self.action'
		# new distribution	
		self.log_policy = LOG_POLICY(self.action_dist_mu, self.action_dist_logstd, self.action)
		# old distribution
		self.log_old_policy = LOG_POLICY(self.old_action_dist_mu, self.old_action_dist_logstd, self.action)

		# Take exponential to log policy distribution
		'''
			Equation (14) in paper
			Contribution of a single s_n : Expectation over a~q[(new policy / q(is)) * advantace_old]
			sampling distribution q is normally old policy
		'''
		policy_ratio = tf.exp(self.log_policy - self.log_old_policy)
		surr_single_state = tf.reduce_mean(policy_ratio * self.advantage)
		# Average KL divergence and shannon entropy 
		kl = GAUSS_KL(self.old_action_dist_mu, self.old_action_dist_logstd, self.action_dist_mu, self.action_dist_logstd) / self.args.batch_size
		ent = GAUSS_ENTROPY(self.action_dist_mu, self.action_dist_logstd) / self.args.batch_size

		self.losses = [surr_single_state, kl, ent]
		tr_vrbs = tf.trainable_variables()
		for i in len(tr_vrbs):
			print(i.op.name)

		'''
			Compute a search direction using a linear approx to objective and quadratic approx to constraint
			=> The search direction is computed by approximately solving 'Ax=g' where A is FIM
				Quadratic approximation to KL divergence constraint
		'''
		# Maximize surrogate function over policy parameter 'theta'
		self.pg = flatgrad(surr_single_state, tr_vrbs)
		# KL divergence where first argument is fixed
		# First argument would be old policy parameters, so keep it constant
		kl_first_fixed = GAUSS_KL_FIRST_FIX(self.action_dist_mu, self.action_dist_logstd) / self.args.batch_size
		# Gradient of KL divergence
		first_kl_grads = tf.gradients(kl_first_fixed, tr_vrbs)
		# Vectors we are going to multiply
		self.flat_tangent = tf.placeholder(tf.float32, [None])
		tangent = list()
		start = 0
		for vrbs in tr_vrbs:
			variable_size = np.prod(vrbs.get_shape().as_list())
			param = tf.reshape(self.flat_tangent[start:(start+variable_size)], vrbs.get_shape())
			tangent.append(param)
			start += variable_size
		'''
			Gradient of KL with tangent vector
			gradient_w_tangent : list of KL_prime*y for each variables  
		'''
		gradient_w_tangent = [tf.reduce_sum(kl_g*t) for (kl_g, t) in zip(first_kl_grads, tangent)]
		'''
			Get derivative of KL_prime*y : [dKL/dx1, dKL/dx2...]
			Returns : [d2KL/dx1dx1+d2KL/dx1dx2..., d2KL/dx1dx2+d2KL/dx2dx2..., ...]
			So get second derivative of KL divergence * y for each variable => y->JMJy
			Use it at computing fisher vector product	
		'''
		self.fim = flatgrad(gradient_w_tangent, tr_vrbs)
		# Get actual paramenter value
		self.get_value = GetValue(self.sess, tr_vrbs)
		# To set parameter values
		self.set_value = SetValue(self.sess, tr_vrbs)

	
		self.sess.run(tf.global_variable_initializer())		


	def train(self):
		batch_path = self.rollout()

		for each_path in batch_path:
			# Value function to calculate advantage
			each_path["Baseline"]


		# Put all paths in batch in a numpy array to feed to network as [batch size, action/observation size]
		# Those batches come from old policy before update theta 
		action_dist_mu = np.squeeze(np.concatenate([each_path["Action_mu"] for each_path in batch_path])
		action_dist_logstd = np.squeeze(np.concatenate([each_path["Action_logstd"] for each_path in batch_path])
		observation = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path])
		action = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path])
		
		feed_dict = {self.obs : , self.action : , self.advantage : , self.old_action_dist_mu : , self.old_action_dist_logstd : }

		# Computing fisher vector product : FIM * (policy gradient), y->Ay=JMJy
		def fisher_vector_product(gradient):
			feed_dict[self.fim] = gradient
			return self.sess.run(self.fim, feed_dict=feed_dict)

		policy_g = self.sess.run(self.pg, feed_dict=feed_dict)
		'''
			Linearize to objective function gives : objective_gradient * (theta-theta_old) = g.transpose * delta
			Quadratize to kl constraint : 1/2*(delta_transpose)*FIM*(delta)
			By Lagrangian => FIM*delta = gradient
		'''
		# Solve Ax = g, where A is FIM and g is gradient of policy network parameter
		# Compute a search direction(delta) by conjugate gradient algorithm
		search_direction = CONJUGATE_GRADIENT(fisher_vector_product, policy_g)

		# KL divergence approximated by 1/2*(delta_transpose)*FIM*(delta)
		# FIM*(delta) can be computed by fisher_vector_product
		# a.dot(b) = a.transpose * b
		kl_approximated = 0.5*search_direction.dot(fisher_vector_product(search_direction))
		# beta
		maximal_step_length = np.sqrt(self.args.kl_constraint / kl_approximated)
		full_step = maximal_step_length * search_direction

		def surrogate(theta):
			self.set_value(theta)
			return self.sess.run(self.losses[0], feed_dict=feed_dict)

		theta_prev = self.get_value()
		# Last, we use a line search to ensure improvement of the surrogate objective and sttisfaction of the KL constraint by manually control valud of parameter
		# Start with the maximal step length and exponentially shrink until objective improves
		new_theta = LINE_SEARCH(surrogate, theta_prev, full_step, self.args.num_backtracking)
		# Update theta	
		self.set_value(new_theta)



	# Make policy network given states
	def build_policy(self, states, name='Policy'):
		with tf.variable_scope(name):
			h1 = linear(states, self.args.hidden_size, name='h1')
			h1_nl = tf.nn.relu(h1)
			h2 = linear(h1_nl, self.args.hidden_size, name='h2')
			h2_nl = tf.nn.relu(h2)
			h3 = linear(h2_nl, self.action_size, name='h3')
			# tf.initializer has to be either Tensor object or 'callable' that takes two arguments (shape, dtype)
			init = lambda shape, dtype : 0.01*np.random.randn(*shape).astype(dtype)
			# [1, action size] since it has to be constant through batch axis, log standard deviation
			action_dist_logstd = tf.get_variable('logstd', initializer=init, shape=[1, self.action_size], dtype=tf.float32)
		
		return h3, action_dist_logstd
		
	def act(self, obs):
		# Need to expand first dimension(batch axis), make [1, observation size]
		obs_expanded = np.expand_dims(obs, 0)
		action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs})
		# Sample from gaussian distribution
		action = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))
		# All shape would be [1, action size]
		print(action)
		return action, action_dist_mu, action_dist_logstd

	def rollout(self):
		if self.args.monitor:
			print('Start monitoring')
			self.env.monitor.start('Rollout', force=True)	
		paths = list()
		timesteps = 0
		self.num_epi = 0
		while timesteps < self.args.timesteps_per_batch:
			self.num_epi += 1
			obs, action, rewards, action_dist_mu, action_dist_logstd = [], [], [], [], []
			prev_obs = self.env.reset()
			for _ in xrange(self.args.max_path_length):
				# Make 'batch size' axis
				prev_obs_expanded = np.expand_dims(prev_obs, 0)
				# Agent take actions and receives sampled action and action distribution parameters
				# All has shape of [1, action size]
				action_, action_dist_mu_, action_dist_logstd_ = self.act(prev_obs)
				# Store observation
				obs.append(prev_obs_expanded)
				action.append(action_)
				action_dist_mu.append(action_dist_mu_)
				action_dist_logstd.append(action_dist_logstd_)
				# Take action 
				next_obs, reward_, done, _ = self.env.step(action_)
				rewards.append(reward_)
				prev_obs = next_obs
				if done:
					# Make dictionary about path, make each element has shape of [None, observation size/action size]
					path = {"Observation":np.concatenate(obs)
							"Action":np.concatenate(action)
							"Action_mu":np.concatenate(action_dist_mu)
							"Action_logstd":np.concatenate(action_dist_logstd)
							# [length,]
							"Reward":np.asarray(rewards)
					paths.append(path)
					print('Path finish')
					break
			timesteps += len(rewards)
		print('%d episodes is collected for batch' % self.num_epi)
		return paths
		



