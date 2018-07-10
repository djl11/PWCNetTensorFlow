import tensorflow as tf
import moviepy.editor as mpy
import tempfile
import cv2
import numpy as np

min_denominator = 1e-12

# Image Warp #
#------------#

def image_warp(im, flow):

    num_batch, height, width, channels = tf.unstack(tf.shape(im))
    max_x = tf.cast(width - 1, 'int32')
    max_y = tf.cast(height - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = tf.reshape(im, [-1, channels])
    flow_flat = tf.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    # The fractional part is used to control the bilinear interpolation.
    flow_floor = tf.to_int32(tf.floor(flow_flat))
    bilinear_weights = flow_flat - tf.floor(flow_flat)

    # Construct base indices which are displaced with the flow
    pos_x = tf.tile(tf.range(width), [height * num_batch])
    grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
    pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]

    # Compute interpolation weights for 4 adjacent pixels
    # expand to num_batch * height * width x 1 for broadcasting in add_n below
    wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
    wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
    wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
    wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

    x0 = pos_x + x
    x1 = x0 + 1
    y0 = pos_y + y
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim1 = width * height
    batch_offsets = tf.range(num_batch) * dim1
    base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
    base = tf.reshape(base_grid, [-1])

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

    return warped

# Visualisation #
#---------------#

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def modify_images_for_vis(x_images, gt_flow, predicted_flow):

    images = list()
    for i in range(2):
        x_image = x_images[i]
        unscaled_predicted_flow = cv2.resize(predicted_flow, (448,384))
        diff_flow = flow_to_image(gt_flow-unscaled_predicted_flow)

        gt_flow_im = flow_to_image(gt_flow)
        predicted_flow_im = flow_to_image(unscaled_predicted_flow)
        diff_flow_im = flow_to_image(diff_flow)

        combined_image = np.concatenate((x_image,gt_flow_im,predicted_flow_im,diff_flow_im),1).astype(np.uint8) # change this to combine all of above

        images.append(combined_image)

    return np.asarray(images)

def convert_array_to_gif_summary(images_arr, tag, fps):

    summary = tf.Summary()

    if len(images_arr.shape) == 5:
        # concatenate batch dimension horizontally
        images_arr = np.concatenate(list(images_arr), axis=-2)
    if len(images_arr.shape) != 4:
        raise ValueError('Tensors must be 4-D or 5-D for gif summary.')
    if images_arr.shape[-1] != 3:
        raise ValueError('Tensors must have 3 channels.')

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(images_arr), fps=fps)
    with tempfile.NamedTemporaryFile() as f:
        filename = f.name + '.gif'
    clip.write_gif(filename, verbose=False, program='ffmpeg')
    with open(filename, 'rb') as f:
        encoded_image_string = f.read()

    image = tf.Summary.Image()
    image.height = images_arr.shape[-3]
    image.width = images_arr.shape[-2]
    image.colorspace = 3  # code for 'RGB'
    image.encoded_image_string = encoded_image_string
    summary.value.add(tag=tag, image=image)
    return summary