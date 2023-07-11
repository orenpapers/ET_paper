from keras.preprocessing.image import img_to_array, load_img
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input

def fn2img(fn):
    # load the image with the required shape

    img = load_img(fn, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    # from a 3D array to a 4D array with the dimensions of [samples, rows, cols, channels], where we only have one sample.
    img = expand_dims(img, axis=0)

    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    return img

def get_video_segment_from_frame_num(x, frames_per_segment = 5):
    return int(x.frame_num / 100 / 5)