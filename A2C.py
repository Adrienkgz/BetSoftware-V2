import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from env import BetEnv
import random
# Définition des paramètres
FILE_PATH_WEIGHT = './WeightAgent/A2CWeightsbest.h5'
GAMMA = 0
LEARNING_RATE = 0.001

# Initialize the environment
env = BetEnv(desired_output='bts', list_leagues_to_exclude=['PL20232024', 'LIGA20232024', 'SerieA20232024', 'BL20232024', 'L120232024', 'EREDIVISIE20212022'], praising_coeff=3.0)

class CartpoleA2C:
    def __init__(self):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=LEARNING_RATE)

    def build_actor(self):
        input_shape_lstm = (5, 7, 24)  # (num_lists, num_matches, num_stats)
        number_parameter_dense = 6 # (num_stats,)
        
        # Define the LSTM layer
        lstm_layer =  tf.keras.layers.SimpleRNN(units=256, return_sequences=True)

        # Define the input layer
        input_layer = tf.keras.layers.Input(shape=input_shape_lstm)

        # Process each list of matches separately
        lstm_outputs = []
        for i in range(4):
            lstm_output = lstm_layer(input_layer[:, i, :, :])
            lstm_outputs.append(lstm_output)

        # Process the list of match inputs in parallel
        x_temp = tf.keras.layers.Dense(32, activation='relu')(input_layer[:, 4, 0, :number_parameter_dense])
        
        
        
        # Concatenate the outputs
        concatenated_output = tf.keras.layers.Concatenate()(lstm_outputs)
        
        concatenated_lstm_and_dense = tf.keras.layers.Concatenate()([concatenated_output[:, 0, :], x_temp])
        
        
        x = Dense(32, activation='relu')(concatenated_lstm_and_dense)
        x = Dense(16, activation='relu')(x)
        
        
        
        # Define the output layer
        output_layer = Dense(3, activation='softmax')(x)

        # Define the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        return model

    def build_critic(self):
        input_shape_lstm = (5, 7, 24)  # (num_lists, num_matches, num_stats)
        number_parameter_dense = 6 # (num_stats,)
        
        # Define the LSTM layer
        lstm_layer =  tf.keras.layers.SimpleRNN(units=256, return_sequences=True)

        # Define the input layer
        input_layer = tf.keras.layers.Input(shape=input_shape_lstm)

        # Process each list of matches separately
        lstm_outputs = []
        for i in range(4):
            lstm_output = lstm_layer(input_layer[:, i, :, :])
            lstm_outputs.append(lstm_output)

        # Process the list of match inputs in parallel
        x_temp = tf.keras.layers.Dense(32, activation='relu')(input_layer[:, 4, 0, :number_parameter_dense])
        
        
        
        # Concatenate the outputs
        concatenated_output = tf.keras.layers.Concatenate()(lstm_outputs)
        
        concatenated_lstm_and_dense = tf.keras.layers.Concatenate()([concatenated_output[:, 0, :], x_temp])
        
        
        x = Dense(32, activation='relu')(concatenated_lstm_and_dense)
        x = Dense(16, activation='relu')(x)
        
        
        outputs = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
        return model

    def get_action(self, state):
        if random.random() < 0.1:
            return env.action_space.sample()
        action_probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action
    
    def get_best_action(self, state):
        if state is None:
            print()
        action_probs = self.actor.predict(state, verbose=0)[0]
        values_actions = self.critic.predict(state)
        action = np.argmax(action_probs)
        return action
    
    def train(self, episodes=1):
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            state = env.reset()
            episode_reward = 0
            done = False
            i = 0
            env.payroll = 100
            while not done and i < 100:
                print(f"\rEpisode {episode + 1}/{episodes} - Step {i + 1}/100 - Payroll: {env.payroll}", end="")
                # On ajoute une dimension pour que le modèle puisse prédire
                state = np.expand_dims(state, axis=0)
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # Compute TD error
                target = reward
                td_error = target - self.critic.predict(state, verbose=0)

                # Update critic
                self.critic_optimizer.minimize(lambda: self.critic_loss(state, target), var_list=self.critic.trainable_variables)

                # Update actor
                self.actor_optimizer.minimize(lambda: self.actor_loss(state, action, td_error), var_list=self.actor.trainable_variables)

                state = next_state
                i += 1
            list_leagues_to_test = ['PL20232024', 'LIGA20232024', 'SerieA20232024', 'BL20232024', 'L120232024']
            self.test(list_league_to_test=list_leagues_to_test)

        self.save_weights()

    def actor_loss(self, state, action, td_error):
        action_probs = self.actor(state)
        action_log_probs = tf.math.log(tf.reduce_sum(action_probs * tf.one_hot(action, self.action_dim), axis=1))
        actor_loss = -tf.reduce_mean(action_log_probs * td_error)
        return actor_loss

    def critic_loss(self, state, target):
        critic_loss = tf.reduce_mean(tf.square(target - self.critic(state)))
        return critic_loss

    def save_weights(self):
        self.actor.save_weights(FILE_PATH_WEIGHT)
        print("Weights saved")

    def load_weights(self):
        if tf.io.gfile.exists(FILE_PATH_WEIGHT):
            self.actor.load_weights(FILE_PATH_WEIGHT)
            print("Weights loaded")
        else:
            print("No weights found")
    
    def test(self, list_league_to_test:list):
        for league in list_league_to_test:
            print(f"\nTesting league {league}")
            local_env = BetEnv(desired_output='bts', only_league_params=league, training_mode=False)
            state = local_env.reset()
            episode_reward = 0
            done = False
            while not done:
                print(f'\rPayroll: {local_env.payroll}', end="")
                # On ajoute une dimension pour que le modèle puisse prédire
                state = np.expand_dims(state, axis=0)
                action = self.get_action(state)
                next_state, reward, done, _ = local_env.step(action)
                episode_reward += reward
                state = next_state
                if state is None:
                    break
            print(f"League {league}: Reward = {episode_reward}")

a2c = CartpoleA2C()
a2c.train(episodes = 100)
list_leagues_to_test = ['PL20232024', 'LIGA20232024', 'SerieA20232024', 'BL20232024', 'L120232024', 'EREDIVISIE20212022']
a2c.test(list_league_to_test=list_leagues_to_test)
