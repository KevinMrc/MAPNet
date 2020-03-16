import os
import re
import cv2
import time
import imageio
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

from model.mapnet import mapnet
from load_data import load_batch, prepare_data

############################# Helper functions #################################

def scores(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    tn = np.sum(np.logical_and(predict == 0, label == 0))
    fp = np.sum(np.logical_and(predict == 1, label == 0))
    fn = np.sum(np.logical_and(predict == 0, label == 1))
    
    intersection = tp
    union = fp + fn + tp

    return intersection, union, tp, tn, fp, fn 
  
def make_mask(im):
    im[im < 0.5] = 0
    im[im >= 0.5] = 1
    return im
  

#################################### Args ######################################

parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=2,
                    help='Number of images in each batch')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='Number of images in each batch')
parser.add_argument('--num_epochs', type=int, default=80,
                    help='Number of epochs to train for')
parser.add_argument('--weighted_loss', type=int, default=1,
                    help="Weigth for positive detection in weighted cross entropy")

# Input size
parser.add_argument('--crop_height', type=int, default=512,
                    help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512,
                    help='Width of cropped input image to network')
parser.add_argument('--channels', type=int, default=3,
                    help="Number of channels in the input image")

# Data augmentation
parser.add_argument('--h_flip', type=bool, default=True,
                    help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=bool, default=True,
                    help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=bool, default=True,
                    help='randomly rotate, the imagemax rotation angle in degrees.')

# Validation
parser.add_argument('--start_valid', type=int, default=0,
                    help='Number of epoch to valid')
parser.add_argument('--valid_step', type=int, default=1,
                    help="Number of step to validation")

# Folder paths
parser.add_argument('--train_img_path', type=str, default='./Retina_Segmentation/image/train/',
                    help="Image train folder")
parser.add_argument('--train_mask_path', type=str, default='./Retina_Segmentation/mask/train/',
                    help="Mask train folder")
parser.add_argument('--test_image_path', type=str, default='./Retina_Segmentation/image/test/',
                    help="Image test folder")
parser.add_argument('--test_mask_path', type=str, default='./Retina_Segmentation/mask/test/',
                    help="Mask test folder")

parser.add_argument('--checkpoint_load_path', type=str, default='./checkpoint/',
                    help="Folder to load checkpoints from")
parser.add_argument('--checkpoint_save_path', type=str, default='./checkpoint/',
                    help="Folder where to save checkpoints")

parser.add_argument('--save_mask_train', action='store_true',
                    help="Save a mask during training")
parser.add_argument('--no_load', action='store_true',
                    help="Don't load checkpoints")

args = parser.parse_args()


# Load data
num_images = []
train_img, train_label, valid_img, valid_lab = prepare_data(
    args.train_img_path, args.train_mask_path, args.test_image_path, args.test_mask_path)
num_batches = len(train_img) // (args.batch_size)

# Placeholders
img = tf.placeholder(
    tf.float32, [None, args.crop_height, args.crop_width, args.channels])
is_training = tf.placeholder(tf.bool)
label = tf.placeholder(
    tf.float32, [None, args.crop_height, args.crop_height, 1])
pred = mapnet(img, is_training)
pred1 = tf.nn.sigmoid(pred)

# Loss and optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    sig = tf.nn.weighted_cross_entropy_with_logits(
        labels=label, logits=pred, pos_weight=args.weighted_loss)
    sigmoid_cross_entropy_loss = tf.reduce_mean(sig)
    train_step = tf.train.AdamOptimizer(
        args.learning_rate).minimize(sigmoid_cross_entropy_loss)
saver = tf.train.Saver(var_list=tf.global_variables())


def load():
    """
    Load checkpoints
    """
    print("Reading checkpoints...")
    checkpoint_dir = args.checkpoint_load_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("Checkpoint {} read successfull".format(ckpt_name))
        return True, counter
    else:
        print("Checkpoint not found")
        return False, 0


def train():
    # Initialize
    tf.global_variables_initializer().run()
    train_iter, train_loss, loss_tmp = [], [], []
    IOU = 0.65
    
    # Load checkpoints
    if not args.no_load:
        could_load, checkpoint_counter = load()
        if could_load:
            start_epoch = (int)(checkpoint_counter / num_batches)
            start_batch_id = checkpoint_counter - start_epoch * num_batches
            counter = checkpoint_counter
            print("Checkpoint Load Successed")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print("Training from scratch...")
    else:
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        print("Training from scratch...")

    # Print info
    print(f"Total train image: {len(train_img)}")
    print(f"Total validate image: {len(valid_img)}")
    print(f"Total epoch: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image shape: {args.crop_width} x {args.crop_height} x {args.channels}")
    print("Data Argument:")
    print(f"h_flip: {args.h_flip}")
    print(f"v_flip: {args.v_flip}")
    print(f"rotate: {args.rotation}")

    # Epoch
    for i in range(start_epoch, args.num_epochs):
        epoch_time = time.time()
        id_list = np.random.permutation(len(train_img))
        
        # Batch
        for j in range(start_batch_id, num_batches):
            img_d, lab_d = [], []
            for ind in range(args.batch_size):
                id = id_list[j * args.batch_size + ind]
                img_d.append(train_img[id])
                lab_d.append(train_label[id])

            # Load
            x_batch, y_batch = load_batch(
                img_d, lab_d, shape=args.crop_height, channels=args.channels, 
                h_flip=args.h_flip, vflip=args.v_flip, rotation=args.rotation)

            feed_dict = {img: x_batch,
                         label: y_batch,
                         is_training: True
                         }
            # Train
            _, loss, pred1 = sess.run(
                [train_step, sigmoid_cross_entropy_loss, pred], feed_dict=feed_dict)

            # Save mask
            if args.save_mask_train and (j % 100 == 0):
                Image.fromarray(np.squeeze(make_mask(pred1[1]))*255.).convert("L").save('result_thresh_train.png')
                
            # Print loss
            loss_tmp.append(loss)
            if (counter % 100 == 0):
                tmp = np.median(loss_tmp)
                train_iter.append(counter)
                train_loss.append(tmp)
                print('Epoch', i, '|Iter', counter, '|Loss', tmp)
                loss_tmp.clear()
            counter += 1
        start_batch_id = 0
        print('Time:', time.time() - epoch_time)

        # saver.save(sess, './checkpoint/model.ckpt', global_step=counter)

        # Validation
        if i >= args.start_valid:
            if (i - args.start_valid) % args.valid_step == 0:
              val_iou, precision, recall, f1, accuracy, specificity = validation()
              print(f"Last IOU value: {IOU:0.3f}")
              print(f"New IOU value: {val_iou:0.3f}")
              print(f'New Precision: {precision:0.3f}, Recall: {recall:0.3f}, F1: {f1:0.3f}, Accuracy: {accuracy:0.3f}, Specificity: {specificity:0.3f}')

              if val_iou > IOU:
                  print("Save the checkpoint...")
                  saver.save(sess, args.checkpoint_save_path + 'model.ckpt',
                            global_step=counter, write_meta_graph=True)
                  IOU = val_iou

    saver.save(sess, args.checkpoint_save_path + 'model.ckpt', global_step=counter)

def validation():
    print("Start validation...")
    inter, unin, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    for j in range(0, len(valid_img)):
        # Load image    
        x_batch = imageio.imread(valid_img[j]) / 255.0

        # Save image
        if (j % (len(valid_img)//4) == 0):
            print(f'validation {j}')
            if args.channels == 1:
              Image.fromarray(x_batch*255.).convert("L").save('input_image_valid.png')
            elif args.channels == 3:
              Image.fromarray((x_batch*255).astype('uint8')).save('input_image_valid.png')

        # Reshape
        if args.channels == 1:
            x_batch = x_batch[np.newaxis, :, :, np.newaxis]
        elif args.channels == 3:
            x_batch = x_batch[np.newaxis, :, :, :]

        # Inference
        feed_dict = {img: x_batch,
                     is_training: False
                     }
        predict = sess.run(pred1, feed_dict=feed_dict)
        
         # Mask
        result = np.squeeze(make_mask(predict))
        
        # Get groud truth
        gt_value = make_mask(imageio.imread(valid_lab[j]))
        
        # Save mask and ground truth
        if (j % (len(valid_img)//4) == 0):
            Image.fromarray(result*255.).convert("L").save('ouput_mask_valid.png')
            Image.fromarray(gt_value*255).convert("L").save('ground_truth_mask_valid.png')

        # Scores
        intersection, union, tp_, tn_, fp_, fn_  = scores(gt_value, result)
        tp, tn, fn, fp = tp + tp_, tn + tn_, fn + fn_, fp + fp_
        inter, unin = inter + intersection, unin + union

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)

    return inter*1.0 / unin, precision, recall, f1, accuracy, specificity


with tf.Session() as sess:
    train()
