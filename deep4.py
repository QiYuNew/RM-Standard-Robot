import numpy as np
import tensorflow as tf
import cv2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
 
def network(x):
    x_image = tf.reshape(x, [-1,28,28,1]) #-1 means arbitrary
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #conv1
    h_pool1 = max_pool(h_conv1)                               #max_pool1
 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  #conv2
    h_pool2 = max_pool(h_conv2)                               #max_pool2
 
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #fc1
    
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)               #dropout
    rate = 1 - keep_prob
    h_fc1_drop = tf.nn.dropout(h_fc1, rate)               #dropout
 
    y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #fc2 output
    return y_predict

#num  1     2    3     5    6    7    8    9    0
x1 = [0,    0,   0,   560, 280, 280, 280, 560, 560]
y1 = [0,    560, 280, 280, 280, 560, 0,   560,   0]
x2 = [280,  280, 280, 840, 560, 560, 560, 840, 840]
y2 = [280,  840, 560, 560, 560, 840, 280, 840, 280]
    #       图片上画框            ROI
    # 1 (0,0),(280,280)     [0:280,0:280]
    # 2 (0,560),(280,840)   [560:840,0:280]
    # 3 (0,280),(280,560)   [280:560,0:280]
    # 5 (560,280),(840,560) [280:560,560:840]
    # 6 (280,280),(560,560) [280:560,280:560]
    # 7 (280,560),(560,840) [560:840,280:560]
    # 8 (280,0),(560,280)   [0:280,280:560]
    # 9 (560,560),(840,840) [560:840,560:840]
    # 0 (560,0),(840,280)   [0:280,560:840]

keep_prob = tf.placeholder("float")
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
sess=tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./model_save.ckpt") #load model file must have ./ with tensorflow1.0

frame = cv2.imread('C:/Users/27774/Desktop/opencv/all_rev.png')
cv2.bitwise_not(frame,frame)
i = 0
print('which number do you want:')
num = input()
num = int(num)
#print(num)

for i in range(9):
    roiImg = frame[y1[i]:y2[i],x1[i]:x2[i]]#y1:y2,x1:x2
    img = cv2.resize(roiImg,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_img = img.astype(np.float32)

    netoutput = network(np_img)
    predictions = sess.run(netoutput,feed_dict={keep_prob: 0.5})

    predicts=predictions.tolist() #tensorflow output is numpy.ndarray like [[0 0 0 0]]
    label=predicts[0]
    result=label.index(max(label))
    #print(result)

    if result==num:
        print("show")
        print("Any key to exit!")
        cv2.rectangle(frame,(x1[i],y1[i]),(x2[i],y2[i]),(0,0,255),2)#图，起点(x1,y1)，终点(x2,y2)，颜色，粗细
        cv2.imshow("img", frame)
        cv2.waitKey()
        break
        
cv2.destroyAllWindows()

