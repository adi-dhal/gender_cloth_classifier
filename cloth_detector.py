###################################################################
import sys
import dlib
import cv2
import os
import numpy as np
import tensorflow as tf
##################################################################
		## Variables
##################################################################
images = []
labels = []
n_classes = 50
batch_size = 64
x = tf.placeholder('float', [None, 16384])
y = tf.placeholder('float')
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
detector = dlib.get_frontal_face_detector()
##################################################################
##		CNN Framework
##################################################################
		##Convulation Function
		
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
###################################################################
		##SubSampling Function
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

###################################################################

		## CNN Structure
def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([32*32*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    print x
    x = tf.reshape(x, shape=[-1, 128, 128, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 32*32*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    #fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output
##################################################################################    
    			##CNN Trainer

def train_neural_network(x):
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	hm_epochs = 10
	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.initialize_all_variables())
		for i in range(300):
			for epoch in range(hm_epochs): 
				epoch_loss = 0
				epoch_x, epoch_y = batch_feeder(i)
				print epoch_y.shape		
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
				print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
	
		saver.save(sess,"./cloth/model.ckpt")
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		epoch_x, epoch_y = batch_feeder(301)
		print('Accuracy:',accuracy.eval({x:epoch_x, y:epoch_y}))		

#####################################################################
# Using neural network

def use_cloth_neural_network(input_data):
	print("---------*** using neural network--*")
	prediction = convolutional_neural_network(x)
	#feature = process_image(input_data)	
	feature = input_data
	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.initialize_all_variables())
		saver.restore(sess,"./cloth/model.ckpt")		
		result = (sess.run(tf.argmax(prediction.eval(session=sess,feed_dict={x:feature}),1)))
	
	return result[0]





#######################################################

##feeder function giving batchwise

def batch_feeder(number):
	img = images[(number)*128:(number+1)*128]	
	lab = labels[(number)*128:(number+1)*128]	
	img = np.array(img)
	lab = np.array(lab)
	print img.shape	
	img = np.reshape(img,(128,16384))
	
	return img,lab
	
def process_image(img):
	img = cv2.imread(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rsz = np.array(rsz,dtype='float32')
	##Normalising the image 0-1
	normalizedImg = np.zeros((128, 128),dtype='float32')
	normalizedImg = cv2.normalize(rsz,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
	normalizedImg = np.reshape(normalizedImg,(1,16384))
	return normalizedImg
	
	
####################################################################	
##		Driver Part
####################################################################
##making detector instance

def preprocess1():

	##Extracting files

	filenames = os.listdir(os.path.join(os.getcwd(),'cloth/processed'))
	filenames = [name for name in filenames if name != 'face_detector.py']


	##Testing
	filenames = filenames[0:100000]

	
	for f in filenames :
		try:
			img = cv2.imread('./cloth/processed/'+f)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		## consider image with only one detected face

			
			rsz = np.array(img,dtype='float32')
	
			##Normalising the image 0-1
			normalizedImg = np.zeros((128, 128),dtype='float32')
			normalizedImg = cv2.normalize(rsz,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
		except:
			continue	
			## constructing the images and labels
		images.append(normalizedImg)
		cat = f.split('_')[1].split('.')[0]
		
		lab = np.zeros(50,dtype = float)
		lab[int(cat)-1] = 1.0
		print '2'
		labels.append(lab)
		

if len(sys.argv) == 1:
	preprocess1()		
	train_neural_network(x)
else:
	use_cloth_neural_network(sys.argv[1])





