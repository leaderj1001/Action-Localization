import numpy as np
import cv2
from utils.config import cfg


def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images are already prepared (means subtracted, BGR order, ...). """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    return blob


def prep_im_for_blob(im, pixel_means):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    # im -= pixel_means
    im = cv2.resize(im, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)

    return im


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified scales."""

    blob = np.zeros((64, 400, 400, 3), dtype=np.float32)
    for idx, image_path in enumerate(roidb[0]['image_path']):
        processed_ims = []
        im = cv2.imread(image_path)
        im = prep_im_for_blob(im, cfg.PIXEL_MEANS)
        processed_ims.append(im)

        # Create a blob to hold the input images
        frame = im_list_to_blob(processed_ims)

        blob[idx, :, :, :] = frame

    return blob
