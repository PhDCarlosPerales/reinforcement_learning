from learners.Learner import Learner


class QLearner(Learner):

    def update(self,game, old_state,action_taken, reward_action_taken, new_state, new_action,reached_end):
        idx_action_taken =list(game.action_space).index(action_taken)

        actual_q_value_options = self._q_table[old_state[0], old_state[1]]
        actual_q_value = actual_q_value_options[idx_action_taken]


        future_q_value_options = self._q_table[new_state[0], new_state[1]]
        future_max_q_value = reward_action_taken  +  self.discount_factor*future_q_value_options.max()
        if reached_end:
            future_max_q_value = reward_action_taken #maximum reward

        self._q_table[old_state[0], old_state[1], idx_action_taken] = actual_q_value + \
                                              self.learning_rate*(future_max_q_value -actual_q_value)



