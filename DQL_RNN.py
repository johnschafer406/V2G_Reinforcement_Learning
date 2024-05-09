
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
import random
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, sequence_length, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length= sequence_length
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.0 # discount rate tweak
        self.epsilon = .2  # exploration rate Tweak, no decay in this version
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # tweak
        self.learning_rate = 0.1
        
        if model is None:
            self.model = self._build_model()
        else:
            self.model = model
    """
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #LSTM for temporal dependicies of data
        model = tf.keras.Sequential()
        model.add(layers.LSTM(50, input_shape=(self.sequence_length, self.state_size), return_sequences=True))
        model.add(layers.LSTM(50, return_sequences=False))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))
        
        return model
        """
    def _build_model(self):
        model = tf.keras.Sequential()
        # First RNN layer with return_sequences=True to pass sequences to the next RNN layer
        model.add(layers.SimpleRNN(50, input_shape=(self.sequence_length, self.state_size), return_sequences=True))
        # Second RNN layer with return_sequences=False to flatten the output
        model.add(layers.SimpleRNN(50, return_sequences=False))
        model.add(layers.Dense(24, activation='relu'))
        # Output layer for action predictions
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))

        return model



    def act(self, state_history):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        lstm_input = np.expand_dims(state_history, axis=0)
        act_values = self.model.predict(lstm_input, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def learn(self, state_history, action, reward, future_state, done):
        next_state_history = np.append(state_history[1:], [future_state], axis=0)  # make sequnce for future state

        lstm_input = np.expand_dims(state_history, axis=0)
        next_lstm_input = np.expand_dims(next_state_history, axis=0)

        target = reward
        if not done:
            target = (reward + self.gamma * np.amax(self.model.predict(next_lstm_input, verbose=0)[0]))

        target_f = self.model.predict(lstm_input, verbose=0)
        target_f[0][action] = target
        self.model.fit(lstm_input, target_f, epochs=1, verbose=0)

