import numpy as np


class Learner:
    def __init__(self,game):
        position = list(game.positions_space)
        position.append(len(game.action_space))
        self._q_table=np.zeros(position)
        self.discount_factor = 0.1
        self.learning_rate = 0.1
        self.ratio_explotacion = 0.9

    def get_next_step(self, state, game):
        next_step=np.random.choice(list(game.action_space))
        if np.random.uniform() <= self.ratio_explotacion:
            # take max unless tie, then random
            idx_action = np.random.choice(np.flatnonzero(
                    self._q_table[state[0],state[1]] == self._q_table[state[0],state[1]].max()
                ))
            next_step = list(game.action_space)[idx_action]

        return next_step

    def print_policy(self):
        for row in np.round(self._q_table,1):
            for column in row:
                print('[', end='')
                for value in column:
                    print(str(value).zfill(5), end=' ')
                print('] ', end='')
            print('')
