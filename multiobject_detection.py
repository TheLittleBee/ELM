import sys

reload(sys)
sys.setdefaultencoding('utf8')

from skimage import io,draw,color
from skimage.transform import resize
from skimage.feature import hog,local_binary_pattern
from selectivesearch import selective_search
import tensorflow as tf
from rfcelm import RFCELM
from celm import CELM
from sklearn.externals import joblib
import numpy as np

key = ['cat','car', 'bike','person','dog']
win_chosen = {'car': [], 'bike': [],'person':[],'dog':[],'cat':[]}
drawcolor={'car':[255,0,0],'bike':[0,255,0],'person':[0,0,255],'dog':[255,255,0],'cat':[0,255,255]}


def slide_win(img, step_size, scale_n=1, min_win=(100, 100)):
	win_x = np.random.randint(min_win[0], img.shape[0], scale_n)
	win_y = np.random.randint(min_win[1], img.shape[1], scale_n)
	for i in range(scale_n):
		# slide a window across the image
		for x in xrange(0, img.shape[0] - win_x[i], step_size):
			for y in xrange(0, img.shape[1] - win_y[i], step_size):
				# yield the current window
				yield (x, y, img[y:y + win_y[i],x:x + win_x[i]])


def overlapping_area(detection_1, detection_2):
	'''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
	# Calculate the x-y co-ordinates of the
	# rectangles
	x1_tl = detection_1[0]
	x2_tl = detection_2[0]
	x1_br = detection_1[0] + detection_1[3]
	x2_br = detection_2[0] + detection_2[3]
	y1_tl = detection_1[1]
	y2_tl = detection_2[1]
	y1_br = detection_1[1] + detection_1[4]
	y2_br = detection_2[1] + detection_2[4]
	# Calculate the overlapping Area
	x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
	y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
	overlap_area = x_overlap * y_overlap
	area_1 = detection_1[3] * detection_2[4]
	area_2 = detection_2[3] * detection_2[4]
	total_area = area_1 + area_2 - overlap_area
	return overlap_area / float(total_area)


def nms(detections, threshold=.3):
	'''
    This function performs Non-Maxima Suppression.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    '''
	if len(detections) == 0:
		return []
	# Sort the detections based on confidence score
	detections = sorted(detections, key=lambda detections: detections[2], reverse=True)
	# Unique detections will be appended to this list
	new_detections = []
	# Append the first detection
	new_detections.append(detections[0])
	# Remove the detection from the original list
	del detections[0]
	# For each detection, calculate the overlapping area
	# and if area of overlap is less than the threshold set
	# for the detections in `new_detections`, append the
	# detection to `new_detections`.
	# In either case, remove the detection from `detections` list.
	for index, detection in enumerate(detections):
		for new_detection in new_detections:
			if overlapping_area(detection, new_detection) > threshold:
				del detections[index]
				break
		else:
			new_detections.append(detection)
			del detections[index]
	return new_detections


image = io.imread('cars/carsgraz_001.bmp')
# imagelbl,regions=selective_search(image,scale=1, sigma=0.8, min_size=10)
img = color.rgb2gray(image)
# candidates = set()
# for r in regions:
# 	# excluding same rectangle (with different segments)
# 	if r['rect'] in candidates:
# 		continue
# 	# excluding regions smaller than 2000 pixels
# 	if r['size'] < 2000:
# 		continue
# 	candidates.add(r['rect'])

# sess = tf.InteractiveSession()
# elm = RFCELM(sess, 1536, 15876, 1500, 6)
# elm1 = RFCELM(sess, 1536, 128 * 128, 1500, 6)
# elm = CELM(sess, 1536, 15876, 1500, 6)
# elm1 = CELM(sess, 1536, 128 * 128, 1500, 6)
# saver = tf.train.Saver()
# ckpt = tf.train.get_checkpoint_state('model/lbp/')
# if ckpt and ckpt.model_checkpoint_path:
# 	saver.restore(sess, ckpt.model_checkpoint_path)
clf=joblib.load('model/svm/car.txt')
i=0
for (x, y, window) in slide_win(img, 10):
# for(x,y,w,h)in candidates:
	i+=1
	# window = img[y:y+h,x:x+w]
	clone = resize(window, [128, 128], mode='wrap')

	# feature_extract
	f_hog,fv = hog(clone,visualise=True)
	f_lbp = local_binary_pattern(clone,8,1.,method='ror')/255.
	# io.imshow(fv)
	# io.show()
	# classifier
	pred=clf.predict(f_hog.reshape(1,-1))
	clfdecision=clf.decision_function(f_hog.reshape(1,-1))
	# y0=elm.test(f_hog[np.newaxis],trained=True)
	# y1=elm1.test(f_lbp[np.newaxis].reshape([1,128*128]),trained=True)
	# pred = np.argmax(0.7*y0+0.3*y1).astype(int)
	# clfdescion = np.max(0.7*y0+0.3*y1)
	# if pred != 5 :
	if pred ==1:
		print clfdecision
		win_chosen[key[pred]].append((x, y, clfdecision, window.shape[1], window.shape[0]))
print i
clone = image.copy()
for key in win_chosen:
	win_chosen[key] = nms(win_chosen[key])
	if len(win_chosen[key]) != 0:
		for (x, y, d, w, h) in win_chosen[key]:
			#visualize op
			r,c=draw.polygon_perimeter([y,y+h,y+h,y],[x,x,x+w,x+w])
			draw.set_color(clone,[r,c],drawcolor[key])

print win_chosen
io.imshow(clone)
io.show()