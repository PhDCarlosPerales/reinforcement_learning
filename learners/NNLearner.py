import random

from keras import Sequential
from keras.layers import Dense, np
from keras.optimizers import Adam


class NNLearner():
    def __init__(self,game):
        self.state_size = (np.prod(game.positions_space),)

        self.grid_size = np.prod(game.positions_space)
        self.action_size = len(game.action_space)
        self.memory = list()#deque(maxlen=2000)
        self.max_memory = 2000
        self.learning_rate = 0.1    # discount rate
        self.ratio_explotacion = 0.8#0.95# 0.8  # exploration rate
        self.explotacion_min = 0.05
        self.explotacion_decay = 0.995
        self.model_learning_rate = 0.001
        self.discount_factor = 0.1
        self.model = self._crear_modelo()
        self.game = game
        self.update_iteration=0

    def _crear_modelo(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.model_learning_rate))


        return model

    def update(self, game, old_state, action_taken, reward_action_taken, new_state, new_action, reached_end):
        self.update_iteration+=1
        if reached_end or self.update_iteration>=999:  # entrenamos este epoch
            self.remember(old_state, action_taken, reward_action_taken,
                          new_state, reached_end)
            self.aprendizaje(min(100,len(self.memory)))
            self.update_iteration=0

        else:
            # guardamos entrenamiento
            self.remember(old_state, action_taken, reward_action_taken,
                          new_state, reached_end)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self.prepare_state(state), self.prepare_action(action), reward, self.prepare_state(next_state), done))
        if len(self.memory) > self.max_memory:
            del self.memory[0] # se quita el primero

    def get_next_step(self, state, game):
        next_step = np.random.choice(list(game.action_space))
        if np.random.uniform() <= self.ratio_explotacion:
            q = self.model.predict(self.prepare_state(state))
            idx_action = np.argmax(q[0])
            next_step = list(game.action_space)[idx_action]
        return next_step

    def aprendizaje(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        minibatch.append(self.memory[-1]) # final estate
        for state, action, reward, next_state, done in minibatch:
            actual_q_value_options = self.model.predict(state)
            actual_q_value = actual_q_value_options[0][action]

            future_max_q_value = reward
            if not done:
                future_max_q_value = reward + self.discount_factor*np.amax(self.model.predict(next_state)[0])

            actual_q_value_options[0][action] = actual_q_value + self.learning_rate*(future_max_q_value - actual_q_value)
            self.model.fit(state, actual_q_value_options, epochs=1, verbose=0)

        # change exploration rate
        if 1-self.ratio_explotacion > self.explotacion_min:
            ratio_exploracion= 1-self.ratio_explotacion
            ratio_exploracion*= self.explotacion_decay
            self.ratio_explotacion = 1- ratio_exploracion

    def prepare_state(self, state):  # 2d to list
        array_1d = np.zeros(self.grid_size)
        idx = state[0] * self.game.positions_space[0] + state[1]
        array_1d[idx] = 1
        return (array_1d.reshape((1, -1)))

    def prepare_action(self, action_taken):
        return (list(self.game.action_space).index(action_taken))

    def print_policy(self):
        array_1d = np.zeros(self.grid_size)
        tabs=0
        for i in range(0,self.grid_size):
            array_1d[i] = 1
            sep = ' '
            if not ((i + 1) % self.game.positions_space[1]):
                # tabs += 1
                sep = '\n'
            print(self.model.predict(array_1d.reshape((1, -1))),end=sep)
            array_1d[i] = 0