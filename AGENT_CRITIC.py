import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import legacy as lg
from tensorflow.keras.optimizers import Adam
import random

class AGENT_CRITIC:
    def __init__(self, state_size, action_size, sequence_length, actormodel=None, criticmodel=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = .99  # Discount rate
        self.learning_rate = 0.001
        self.epsilon = 0  # Exploration factor
        self.epsilon_decay=.90
        self.optimizer = Adam(learning_rate=self.learning_rate)

        #WORKS WITH GAMMA=.99, lr=.001, epsilon=.1
        
        if criticmodel is None:
            self.critic = self.build_critic()
        else:
            self.critic = criticmodel

        if actormodel is None:
            self.actor = self.build_actor()
        else:
            self.actor = actormodel
    def build_actor(self):
        inputs = Input(shape=(self.sequence_length, self.state_size))
        x = SimpleRNN(50, return_sequences=True)(inputs)
        x = SimpleRNN(50)(x)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self):
        inputs = Input(shape=(self.sequence_length, self.state_size))
        x = SimpleRNN(50, return_sequences=True)(inputs)
        x = SimpleRNN(50)(x)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def custom_loss(self, delta):
        def loss(y_true, y_pred):
            epsilon = 1e-8
            log_probs = y_true * tf.math.log(tf.clip_by_value(y_pred, epsilon, 1.0))
            return -tf.reduce_mean(log_probs * delta)
        return loss


    def act(self, state):
        # Check if we should explore randomly
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape the state to match the expected input format of the model
        state = state[np.newaxis, :]  # Adds an extra dimension to the state, making it (1, state_dimension)
        
        # Predict action probabilities using the actor model
        probabilities = self.actor.predict(state, verbose=0)[0]
    
        # Choose an action based on the probability distribution
        return np.random.choice(self.action_size, p=probabilities)

    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            y_pred = self.actor(states, training=True)
            delta = self.calculate_delta(states, rewards, next_states, dones)
            action_masks = tf.one_hot(actions, self.action_size)
            loss = self.custom_loss(delta)(action_masks, y_pred)

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        return loss



    def learn(self, state_history, action, reward, next_state, done):
        if len(state_history) < self.sequence_length:
            return
        
        # Prepare the inputs for the actor and critic
        lstm_input = np.expand_dims(state_history, axis=0)
        next_state_history = np.append(state_history[1:], [next_state], axis=0)
        next_lstm_input = np.expand_dims(next_state_history, axis=0)

        # Compute values for current and next states using the critic
        with tf.GradientTape() as tape:
            critic_value = self.critic(lstm_input, training=True)
            critic_value_next = self.critic(next_lstm_input, training=True)
            
            # Calculate target and delta
            target = reward + (0 if done else self.gamma * tf.squeeze(critic_value_next))
            delta = target - tf.squeeze(critic_value)

            # Run the actor model and calculate the custom loss
            y_pred = self.actor(lstm_input, training=True)
            action_masks = tf.one_hot([action], self.action_size)
            loss = self.custom_loss(delta)(action_masks, y_pred)

        # Compute gradients and apply them for both actor and critic
        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))

        # Optionally, update the critic separately if needed
        self.critic.fit(lstm_input, np.array([[target]]), verbose=0, batch_size=1)