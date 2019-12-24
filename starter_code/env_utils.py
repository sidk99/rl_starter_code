import cv2

def render(env, scale):
    frame = env.render(mode='rgb_array')
    if frame is not None:
        h, w, c = frame.shape
        frame = frame.reshape(w, h, c)
        frame = cv2.resize(frame, dsize=(int(h*scale), int(w*scale)), interpolation=cv2.INTER_CUBIC)
        return frame
