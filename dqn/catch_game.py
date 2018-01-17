# coding=utf-8

"""
ref:https://github.com/JannesKlaas/sometimes_deep_sometimes_learning
"""

import numpy as np


# translate the actions to human readable words
translate_action = ["Left", "Stay", "Right", "Create Ball", "End Test"]
# size of the game field
grid_size = 10

num_actions = 3

class Catch(object):
    """
    Class catch is the actual game.
    In the game, fruits, represented by white tiles, fall from the top.
    The goal is to catch the fruits with a basked (represented by white tiles, this is deep learning, not game design).
    """

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size - 1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,) * 2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2] - 1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size - 1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size - 1, size=1)
        m = np.random.randint(1, self.grid_size - 2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]

    def get_action_name(self, action):
        return translate_action[action]

    def to_canvas(self, input):
        print(input.shape)
        canvas = input[0]
        canvas = canvas.reshape((grid_size, grid_size))
        canvas = [map(lambda x: 'O' if x < 1 else 'X', row) for row in canvas]
        return canvas


