# imports..
import torch
import numpy as np
from PIL import Image


def process_image(filepath):
    '''
        Returns the image as a tensor..
    '''

    img_size = 224
    im = Image.open(filepath)

    # finding the smallest dimension,
    # keeping aspect ratio, and resizing..
    w, h = im.size
    if (w < h):
        resize = img_size, int(round(h * (img_size / w)))
    else:
        resize = int(round(w * (img_size / h))), img_size

    im.thumbnail(resize)
    resized_w, resized_h = im.size

    # crop..
    # https://github.com/nkmk/python-tools/blob/3b77c6cbc5ed987da7df61576ef6cae60adbae0c/tool/lib/imagelib.py#L71
    im = im.crop((
        (resized_w - img_size) // 2,
        (resized_h - img_size) // 2,
        (resized_w + img_size) // 2,
        (resized_h + img_size) // 2
    ))

    # convert to np array..
    np_image = np.array(im)
    np_image = np_image / 255

    # normalize array..
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image)


def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
