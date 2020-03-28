import scipy
import scipy.misc
import tensorflow as tf
import os
import numpy as np
import glob
import argparse
from PIL import Image
import PIL
from imageio import imread, imwrite

from model. mapnet import mapnet

parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument('--test_img_path', type=str, default='./dataset_lung/Img/Test/',
                    help="Image test folder")
parser.add_argument('--test_result_path', type=str, default='./dataset_lung/Results/',
                    help="Test results folder")  
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/',
                    help="Checkpoint folder")  
parser.add_argument('--crop_size', type=int, default=512,
                    help='size of cropped input image to network')
parser.add_argument('--channels', type=int, default=3,
                    help="Number of channels in the input image")                    
args = parser.parse_args()


batch_size = 1
img = tf.placeholder(tf.float32, [batch_size, args.crop_size, args.crop_size, args.channels])

test_img = sorted(
    glob.glob(r'{}'.format(args.test_img_path) + r'*.png'))

test_results_path = args.test_result_path

pred = mapnet(img, is_training=False)
pred = tf.nn.sigmoid(pred)
saver = tf.train.Saver(tf.global_variables())


def save():
    tf.global_variables_initializer().run()
    checkpoint_dir = args.checkpoint_dir
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

    for j in range(0, len(test_img)):
        x_batch = test_img[j]
        i = x_batch.split('/')[-1]
        x_batch = imread(x_batch) 
        x_batch = np.double(np.array(Image.fromarray(x_batch).resize(size = (args.crop_size, args.crop_size),resample = PIL.Image.BILINEAR)))
        x_batch /= 255.0
        x_batch = np.expand_dims(x_batch, axis=0)
        if args.channels == 1:
            feed_dict = {img: np.expand_dims(x_batch, axis=3)}
        else:
            feed_dict = {img: x_batch}
        predict = sess.run(pred, feed_dict=feed_dict)
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 1
        result = np.squeeze(predict)
        i = i.split('.')[0]
        imwrite(test_results_path + '{}.png'.format(i), (result * 255.0).astype(np.uint8))

with tf.Session() as sess:
    save()

