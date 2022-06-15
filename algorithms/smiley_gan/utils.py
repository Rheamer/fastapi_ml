import numpy as np
import cv2


def finalize_generated_img(img):
    img = (255 - ((img + 1) * 127.5)).astype(np.uint8)
    img = img.reshape((28, 28))
    img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_NEAREST)
    return img


def downsize(img):
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    img = img.reshape(28, 28, 1).astype("float32") / 127.5 - 1
    return img

