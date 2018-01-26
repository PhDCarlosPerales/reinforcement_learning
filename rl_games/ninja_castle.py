import numpy as np


class NinjaCastle:
    def __init__(self):
        self.action_space = {'Arriba': np.array([-1, 0]),
                             'Abajo':np.array([1,0]),
                             'Izquierda':np.array([0,-1]),
                             'Derecha':np.array([0,1])}
        self._step_penalization = -1
        self._rewards = np.array([[0, 0, -100, 0],
                                  [0, 0, -100, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 100]])
        self._final_state = np.array([3,3])
        self.state = np.array([0,0]) #where is the ninja
        self.total_reward = 0
        self.positions_space=self._rewards.shape #inside reward

    def reset(self):
        self.total_reward = 0
        self.state = [0,0]
        return self.state

    def render(self):
        place=self._rewards.astype(str)
        place[self.state[0],self.state[1]] = 'X'
        print(place)

    def step(self, action):
        self._apply_action(action)
        done = np.array_equal(self.state,self._final_state) # final
        info = ""
        reward = self._rewards[self.state[0], self.state[1]]
        reward+= self._step_penalization
        self.total_reward+= reward
        return self.state,reward , done, info

    def _apply_action(self, action):
        self.state+=self.action_space[action]
        if self.state[0] > self._rewards.shape[0]-1:
            self.state[0] = self._rewards.shape[0] - 1
        elif self.state[0] < 0:
            self.state[0] = 0

        if self.state[1] > self._rewards.shape[1] - 1:
            self.state[1] = self._rewards.shape[1] - 1
        elif self.state[1] < 0:
            self.state[1] = 0

