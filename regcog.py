

import _thread
import cv2
#import cv2.cv as cv
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten
import skimage.data
import skimage.transform
import pandas as pd

sign_names = pd.read_csv("signnames.csv")

#path = "/home/pi/traffic-lenet5/new-img/"
path = "D:/luanvan/roadsign_classifier_Lenet5/new_img/img1"

#traffic_sign = cv2.CascadeClassifier('traffic_cascade.xml')

def load_data(data_dir):
	images = []
	file_names = [os.path.join(data_dir,f)for f in os.listdir(data_dir) if f.endswith(".jpg")]
	for f in file_names:
		images.append(skimage.data.imread(f))
	return images


images = load_data(path)


#images = np.asarray(images)

saver = tf.train.import_meta_graph("lenet/EdLeNet_Color_Norm_5x5_Dropout_0.40.chkpt.meta")


graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)
x = graph.get_tensor_by_name("EdLeNet_Color_Norm_5x5_Dropout_0.40/x:0")
y = graph.get_tensor_by_name("EdLeNet_Color_Norm_5x5_Dropout_0.40/y:0")
prediction = graph.get_tensor_by_name("EdLeNet_Color_Norm_5x5_Dropout_0.40/pre:0")
dropout_placeholder_conv = graph.get_tensor_by_name("EdLeNet_Color_Norm_5x5_Dropout_0.40/drop_conv:0")
dropout_placeholder_fc = graph.get_tensor_by_name("EdLeNet_Color_Norm_5x5_Dropout_0.40/drop_fc:0")
#prediction = graph.get_tensor_by_name("prediction:0")



with tf.Session() as sess:
	saver.restore(sess,tf.train.latest_checkpoint("lenet/./"))
	#x = graph.get_tensor_by_name("x:0")
	#prediction = graph.get_tensor_by_name("prediction:0")
	for i,img in enumerate(images):
		img = skimage.transform.resize(img,(32,32,3), mode = 'constant')
		pred = sess.run(prediction,feed_dict = {x: np.array([img]),dropout_placeholder_conv: 1.0,dropout_placeholder_fc: 1.0 })
		plt.subplot(6,3,i+1)
		plt.subplots_adjust(hspace=0.3)
		plt.axis("off")
		plt.title("Prediction: " +str(sign_names["SignName"][pred[0]]))
		plt.imshow(img)
	plt.show()


'''

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
#cap.set(cv.CV_CAP_PROP_FRAME_WIDTH,640);
#cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT,480);
def transform_image(frame):
    return skimage.transform.resize(frame,(32,32),mode = "constant")
while (True):
    ret,frame = cap.read()

    #signs = traffic_sign.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in signs:
        print("Ddddddddxxdfxdfxdfxdf")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
    #frame = np.asarray(frame)
	#print(frame)
    img = _thread.start_new_thread(transform_image,(frame,))
    print ("xxx:" )
    print(img)
    #img = skimage.transform.resize(frame,(32,32),mode="constant")
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint("lenet/./"))
        pred = sess.run(prediction, feed_dict={x: np.array([img]), dropout_placeholder_conv: 1.0,
											   dropout_placeholder_fc: 1.0})

    cv2.putText(frame,str(sign_names["SignName"][pred[0]]),(100,100) ,font, 2 ,(0,0,255), 3 )
    cv2.imshow ("traffic",frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

'''