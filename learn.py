import tensorflow as tf
import numpy as np
import time, os
import gym
from trpo import TRPO

class LEARNER():
	def __init__(self, args, sess):
		self.args = args
		self.sess = sess

		self.env = gym.make(self.args.env_name)
		self.args.max_path_length = self.env.spec.timestep_limit
		self.agent = TRPO(self.args, self.env, self.sess)
		self.saver = tf.train.Saver()
		
	def learn(self):
		train_index = 0
		total_episode = 0
		total_steps = 0
		while True:
			train_index += 1
			start_time = time.time()
			train_log = self.agent.train()
			total_steps += train_log["Total Step"]
			total_episode += train_log["Num episode"]
			self.write_logs(train_index, total_episode, total_steps, start_time, train_log)
			if np.mod(train_index, self.args.save_interval) == 0:
				self.save(train_index)

			if total_steps > self.args.total_train_step:
				break 
			

	def write_logs(self, train_index, total_episode, total_steps, start_time, log_info):
		log_path = os.path.join(self.args.log_dir, self.model_dir+'.csv')
		if not os.path.exists(log_path):
			log_file = open(log_path, 'w')
			log_file.write("Train step\t," + "Surrogate\t," + "KL divergence\t," + "Number of steps trained\t," + "Number of episodes trained\t," + "Episode.Avg.reward\t," + "Elapsed time\n")
		else:
			log_file = open(log_path, 'a')
		print("Train step %d => Surrogate loss : %3.3f, KL div : %3.8f, Number of Episode/steps trained : %d/%d, Episode.Avg.reward : %3.3f, Time : %3.3f" % (train_index, log_info["Surrogate loss"], log_info["KL_DIV"], total_episode, total_steps, log_info["Episode Avg.reward"], time.time()-start_time))
		log_file.write(str(train_index) + '\t,' + str(log_info["Surrogate loss"]) + '\t,' + str(log_info["KL_DIV"]) + '\t,' + str(total_steps) + '\t,' + str(total_episode) + '\t,' + str(log_info["Episode Avg.reward"]) + '\t,' + str(time.time()-start_time)+'\n')
		log_file.flush()


	def save(self, steps):
		model_name = 'TRPO_GAE'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=steps)
		print('Checkpoint saved at %d train step' % steps)

	@property
	def model_dir(self):
		return '{}_{}lambda'.format(self.args.env_name, self.args.lamda)


