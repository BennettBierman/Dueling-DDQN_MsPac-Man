import logging
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

logging.basicConfig(format="[%(levelname)s:%(asctime)s] " "%(message)s", level=logging.INFO)

try:
    import matplotlib.pyplot as plt
    can_render = True
except:
    logging.warning("Cannot import matplotlib; will not attempt to render")
    can_render = False


def validate(model, render=False, nepisodes=1, frame_stack=4):
    assert hasattr(model, "get_action")
    torch.manual_seed(590060)
    np.random.seed(590060)
    model.eval()

    render = render and can_render

    if render:
        nepisodes = 1
        fig, ax = plt.subplots(1, 1)

    total_reward = 0
    steps_alive = []

    for i in range(nepisodes):

        env = FrameStack(
            AtariPreprocessing(gym.make("ALE/MsPacman-v5"), frame_skip=1, scale_obs=True),
            num_stack=frame_stack
        )
        state = env.reset(seed=590060 + i)[0]

        if render:
            im = ax.imshow(state[0], cmap='gray')

        step = 0

        # play until the agent dies or we exceed 50000 observations
        # Now using all 3 lives
        while env.ale.lives() >= 1 and step < 50000:
            action = model.get_action(np.array(state))
            observation, reward, term, trunc, infos = env.step(action)
            total_reward += reward

            if render:
                img = observation[0]
                im.set_data(img)
                fig.canvas.draw_idle()
                plt.pause(0.1)

            state = observation
            step += 1

        steps_alive.append(step)

    logging.info("Steps taken over each of {:d} episodes: {}".format(nepisodes, ", ".join(str(step) for step in steps_alive)))
    logging.info("Total return after {:d} episodes: {:.3f}".format(nepisodes, total_reward))
