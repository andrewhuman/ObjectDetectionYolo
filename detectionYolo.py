import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

# GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

    box_scores = tf.multiply(box_confidence,box_class_probs)
    print(box_scores)

    box_classes = K.argmax(box_scores,axis=-1)
    print(box_classes)

    box_class_scores = K.max(box_scores,axis=-1)
    print(box_class_scores)


    filtering_mask = box_confidence >= threshold
    print(filtering_mask)


    filtering_mask = tf.reshape(filtering_mask,[-1])
    box_class_scores = tf.reshape(box_class_scores,[-1])
    box_classes = tf.reshape(box_classes,[-1])
    boxes = tf.reshape(boxes,[-1,4])
    print(filtering_mask)
    print(box_class_scores)
    print(box_classes)
    print(boxes)


    scores = tf.boolean_mask(box_class_scores,filtering_mask,name='scores_filter')

    classes = tf.boolean_mask(box_classes,filtering_mask,name='classes_filter')
    boxes =  tf.boolean_mask(boxes,filtering_mask,name='boxes_filter')

    return scores, boxes, classes

# GRADED FUNCTION: iou

def iou(box1, box2):

    xi1 = tf.maximum(box1[0],box2[0])
    print(xi1)
    yi1 = tf.maximum(box1[1],box2[1])
    xi2 = tf.minimum(box1[2],box2[2])
    yi2 = tf.minimum(box1[3],box2[3])
    inter_area = tf.abs(tf.multiply(tf.divide(xi2,xi1),tf.divide(yi2,yi1)))

    box1_area = tf.abs(tf.multiply(tf.divide(box1[2],box1[0]),tf.divide(box1[3],box1[1])))
    box2_area = tf.abs(tf.multiply(tf.divide(box2[2],box2[0]),tf.divide(box2[3],box2[1])))
    union_area = tf.divide(tf.add(box2_area,box1_area), inter_area)
    iou = tf.divide(inter_area,union_area)

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):


    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor

    nms_indices = tf.image.non_max_suppression(boxes=boxes,scores=scores,max_output_size=max_boxes,iou_threshold=iou_threshold)
    print(nms_indices)
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)

    return scores, boxes, classes



def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)


    return scores, boxes, classes


