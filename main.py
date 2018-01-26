import numpy as np

from learners.RandomLearner import RandomLearner
from learners.QLearner import QLearner
from learners.NNLearner import NNLearner
from learners.SARSALearner import SARSALearner
from rl_games.ninja_castle import NinjaCastle

if __name__ == "__main__":
    game = NinjaCastle()

    # learner = RandomLearner(game)
    # learner = QLearner(game)
    # learner = SARSALearner(game)
    learner = NNLearner(game)

    max_points= -9999
    first_max_reached = 0
    total_rw=0

    for played_games in range(0,100):#100
        state = game.reset()
        reward, info, done = None, None, None
        iter=0
        while (done != True and iter<10000):#10000
            old_state = np.array(state)
            next_action=learner.get_next_step(state,game)
            state, reward, done, info = game.step(next_action)
            # For sarsa
            next_post_action = learner.get_next_step(state, game)
            learner.update(game,old_state, next_action,reward, state,next_post_action,done)
            # game.render()
            iter+=1

        total_rw+=game.total_reward
        if game.total_reward > max_points:
            max_points=game.total_reward
            first_max_reached = played_games
        print("******** played_games[", played_games, "] Points[", game.total_reward,"]  Steps[", iter, "] MaxPoint[", max_points,"]")

    print('played_games[',played_games,'] puntuacion Total[',total_rw,'] puntuacion máxima[',max_points,'] en[',first_max_reached,']')
    print(game.action_space.keys())
    learner.print_policy()