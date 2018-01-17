# coding=utf-8

import numpy as np

from catch_game import Catch


def train(env, model, exp_replay, epochs, epsilon, num_actions, batch_size,
          verbose=1):
    win_cnt = 0
    win_hist = []
    for e in range(epochs):
        loss = 0.
        env.reset()
        game_over = False
        input_t = env.observe()

        while not game_over:
            # The learner is acting on the last observed game screen
            # input_t is a vector containing representing the game screen
            input_tm1 = input_t

            """
            We want to avoid that the learner settles on a local minimum.
            Imagine you are eating eating in an exotic restaurant. After some experimentation you find
            that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian
            food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
            It's simple: Sometimes, they just don't let you choose but order something random from the menu.
            Maybe you'll like it.
            The chance that your friends order for you is epsilon
            """
            if np.random.rand() <= epsilon:
                # Eat something random from the menu
                action = np.random.randint(0, num_actions, size=1)
            else:
                # Choose yourself
                # q contains the expected rewards for the actions
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            # If we managed to catch the fruit we add 1 to our win counter
            if reward == 1:
                win_cnt += 1

                # Uncomment this to render the game here
            # display_screen(action,3000,inputs[0])

            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t],
                                game_over)

            # Load batch of experiences
            inputs, targets = exp_replay.get_batch(model,
                                                   batch_size=batch_size)

            # train model on experiences
            batch_loss = model.train_on_batch(inputs, targets)

            # print(loss)
            loss += batch_loss
        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {}".format(e,
                                                                            epochs,
                                                                            loss,
                                                                            win_cnt))
        win_hist.append(win_cnt)
    return win_hist


def display_screen(action, points, canvas):
    s = '-------------------------------------------\n'
    s += "Action %s, Points: %d\n" % (action, points)
    s += '\n'.join(['\t'.join(row) for row in canvas])
    print(s)
    import time
    time.sleep(1)


def test(model):
    # This function lets a pretrained model play the game to evaluate how well it is doing
    global last_frame_time

    # Define environment, game
    env = Catch()
    # c is a simple counter variable keeping track of how much we train
    c = 0
    # Reset the last frame time (we are starting from 0)
    last_frame_time = 0
    # Reset score
    points = 0
    # For training we are playing the game 10 times
    for e in range(10):
        loss = 0.
        # Reset the game
        env.reset()
        # The game is not over
        game_over = False
        # get initial input
        input_t = env.observe()
        # display_screen(3,points,input_t)
        c += 1
        while not game_over:
            # The learner is acting on the last observed game screen
            # input_t is a vector containing representing the game screen
            input_tm1 = input_t
            # Feed the learner the current status and get the expected rewards for different actions from it
            q = model.predict(input_tm1)
            # Select the action with the highest expected reward
            action = np.argmax(q[0])
            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            # Update our score
            points += reward
            display_screen(env.get_action_name(action), points,
                           env.to_canvas(input_t))
            c += 1
            print('predict:%s' % q[0])
