import gym
import numpy as np
import random
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from collections import deque

#Setup for TensorFlow 1 compatibility
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

#Define our Q-learning agent.

class QLearningAgent:
    # Agent constructor: Initialize important parameters and TensorFlow placeholders.
    def __init__(self, env, exploration_probability=0.3, learning_rate=0.01, discount_factor=0.85,
                 replay_buffer_size=10000, batch_size=32):
        # Placeholders for state, action, and target in the TensorFlow graph.
        self.state_in = tf.compat.v1.placeholder(tf.int32, shape=[1])
        self.action_in = tf.compat.v1.placeholder(tf.int32, shape=[1])
        self.target_in = tf.compat.v1.placeholder(tf.float32, shape=[1])
        self.learning_rate = learning_rate
        self.exploration_probability = exploration_probability
        self.discount_factor = discount_factor
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.env = env
        self.state = tf.compat.v1.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.compat.v1.one_hot(self.action_in, depth=self.action_size)
        # The neural network representing our Q-table.
        self.q_state = tf.compat.v1.layers.dense(self.state, units=self.action_size)
        self.q_action = tf.compat.v1.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)
        # Weights for the Q-table, initialized with random normal distribution 
        self.q_state_weights = tf.compat.v1.get_variable("q_table/kernel", shape=(self.state_size, self.action_size),
                                                         initializer=tf.compat.v1.keras.initializers.RandomNormal())
        # Loss function for training the Q-network.
        self.loss = tf.reduce_sum(tf.compat.v1.square(self.target_in - self.q_action))
        # The optimizer, Adam in this case
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # TensorFlow session for execution.
        self.sess = tf.compat.v1.Session()

        # Replay buffer for storing past experiences.
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

        # Global initialization of variables in the TensorFlow graph.
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # A function to add experiences to our replay buffer.
    def add_to_replay_buffer(self, experience):
        self.replay_buffer.append(experience)

    # A function to sample experiences from our replay buffer.
    def sample_from_replay_buffer(self):
        return random.sample(self.replay_buffer, self.batch_size)

    # Train the Q-network with replay.
    def train_with_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            # Wait until there are enough samples in the replay buffer.
            return

        # Grab a batch of experiences from the replay buffer.
        batch = self.sample_from_replay_buffer()

        # Training loop
        for experience in batch:
            # Extracting components from the experience tuple.
            current_state, action, next_state, reward, done = experience
            # If the episode is done, set the Q-values for the next state to zero.
            q_next = np.zeros([self.action_size]) if done else self.sess.run(self.q_state,feed_dict={self.state_in: [next_state]})
            # Compute the target Q-value for the Q-learning update.
            q_target = reward + self.discount_factor * np.max(q_next)
            # Create a feed dictionary for the TensorFlow optimizer.
            feed = {self.state_in: [current_state], self.action_in: [action], self.target_in: [q_target]}
            # Update the Q-network parameters with a single step of optimization.
            self.sess.run(self.optimizer, feed_dict=feed)

    # Function to get the action from the agent based on the current state.
    def get_action(self, state):
        # Compute Q-values for the given state using the current Q-network.
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
        # Choose the action with the highest Q-value (greedy action).
        action_greedy = np.argmax(q_state)
        # Choose a random action to encourage exploration.
        action_random = random.choice(range(self.action_size))
        # Randomly choose an action based on exploration probability.
        return action_random if random.random() < self.exploration_probability else action_greedy

    # The main training function, updating the agent based on its experiences.
    def train(self, current_state, action, next_state, reward, done):
        # Take record of the experience
        experience = (current_state, action, next_state, reward, done)
        # Store it in the replay buffer for later use.
        self.add_to_replay_buffer(experience)
        # Train the agent based on past experiences
        self.train_with_replay()
        # Calculate Q-values for the next state.
        q_next = np.zeros([self.action_size]) if done else self.sess.run(self.q_state,
                                                                         feed_dict={self.state_in: [next_state]})
        # Calculate the target Q-value for the current state.
        q_target = reward + self.discount_factor * np.max(q_next)
        # Feed it to the optimizer
        feed = {self.state_in: [current_state], self.action_in: [action], self.target_in: [q_target]}
        self.sess.run(self.optimizer, feed_dict=feed)
        # If the episode is done, decrease exploration
        if done:
            self.exploration_probability = self.exploration_probability * 0.999

    # Destructor for cleaning up TensorFlow-related stuff when the agent is done
    def __del__(self):
        self.sess.close()


#Initialize environment
env = gym.make("Taxi-v3")

# Creating an instance of our adventurous Q-learning agent.
agent = QLearningAgent(env)

# Lists to keep track of our training
episodes = []
list_rewards = []
exploration_probabilities = []

# Training loop
for ep in range(150000):
    # Resetting the environment for a fresh new episode.
    current_state = env.reset()[0]
    done = False
    total_reward = 0

    # Episode loop
    while not done:
        # Let the agent decide its next move
        action = agent.get_action(current_state)
        # Take a step in the environment.
        next_state, reward, done, info, _ = env.step(action)
        # Train the agent based on the experience of this moment.
        agent.train(current_state, action, next_state, reward, done)
        current_state = next_state
        total_reward += reward

        # Retrieve the Q-table (neural network weights) after each episode.
        with tf.compat.v1.variable_scope("q_table", reuse=tf.compat.v1.AUTO_REUSE):
            weights = agent.sess.run(tf.compat.v1.get_variable("kernel"))
        if done:
            exploration_probabilities.append(agent.exploration_probability)
            episodes.append(ep)
            list_rewards.append(total_reward)
            print(f"Episode: {ep}, Total reward: {total_reward}, eps: {agent.exploration_probability}")
            print(weights)

#Write our results to CSV file
csv_data = zip(episodes, list_rewards, exploration_probabilities)
csv_filename = "training_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Episode', 'Total Reward', 'Exploration Probability'])
    csv_writer.writerows(csv_data)
#Plot our diagrams, change depending on range
plt.plot(episodes[75:301], list_rewards[75:301])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode vs. Total Reward')
plt.show()

env.close()