import cv2
from gym_minigrid.wrappers import ImgObsWrapper
from gym.wrappers.time_limit import TimeLimit
from gym.envs.atari.atari_env import AtariEnv
from gym.envs.box2d.lunar_lander import LunarLander


def render(env, scale):
    frame = env.render(mode='rgb_array')

    if frame is not None:
        h, w, c = frame.shape
        if isinstance(env.env, ImgObsWrapper):
            # reshape for minigrid and babyai
            frame = frame.reshape(w, h, c)

        if isinstance(env.env, LunarLander):
            h, w = w, h

        elif isinstance(env.env, AtariEnv):
            h, w = w, h

        frame = cv2.resize(frame, dsize=(int(h*scale), int(w*scale)), interpolation=cv2.INTER_CUBIC)  # for CartPole, Minigrid
        return frame