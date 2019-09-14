import cv2

def render(env, scale):
    frame = env.render(mode='rgb_array')
    h, w = frame.shape[:-1]
    frame = cv2.resize(frame, dsize=(int(h*scale), int(w*scale)), interpolation=cv2.INTER_CUBIC)
    return frame
