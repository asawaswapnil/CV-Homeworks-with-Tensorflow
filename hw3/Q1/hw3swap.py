import sys
sys.path.append('/own_files/cs2770_hw3/darknet_gpu/python')
import darknet as dn
import os
import time

import re
import pdb

imgpath = "/own_files/cs2770_hw3/darknet_gpu/data_test/VOCdevkit/VOC2007/JPEGImages/"
gtpath = "/own_files/cs2770_hw3/darknet_gpu/data_test/VOCdevkit/VOC2007/labels/"
classnames = "/own_files/cs2770_hw3/darknet_gpu/data/voc.names"
root = '/own_files/cs2770_hw3/darknet_gpu' 
dn.set_gpu(0)


#IoU generator
def IoU(pred, g_truth):
		pred = [float(i) for i in pred]
		gtruth = [float(i) for i in g_truth]
		pred_top = (pred[0], pred[1])
		pred_bottom = (pred[0]+pred[2], pred[1]+pred[3])
		gtop = (gtruth[0], gtruth[1])
		gbottom = (gtruth[0]+gtruth[2], gtruth[1]+gtruth[3])
		lowesttop = max(pred_top[0], gtop[0])
		lowestleft = max(pred_top[1], gtop[1])
		lowestbottom = min(pred_bottom[0], gbottom[0])
		lowestright = min(pred_bottom[1], gbottom[1])
		pred_area = (pred[2]+1) * (pred[3]+1)
		g_area = (gtruth[2]+1) * (gtruth[3]+1)
		iou = max(lowestbottom - lowesttop + 1 , 0) * max(lowestright - lowestleft + 1, 0)
		iou = iou/float(pred_area + g_area - iou)
		return iou


def process(x):
	tup = list(x[2])
	tup[0] -= tup[2]/2
	tup[1] -= tup[3]/2
	new_x = []
	new_x.append(x[0])
	new_x.append(tup)
	return new_x
	

def updatemAP(network, network_meta, im):
	imname = imgpath+im
	predict = dn.detect(network, network_meta, imname, thresh=0.5)
	predict =map(process, predict)
	labelname = gtpath+re.split('\.', im)[0]+'.txt'
	#create ground truth array
	g_file = open(labelname, 'r')
	gfile = g_file.readlines()
	gt = dict()
	for row in gfile:
		found = re.split('\s', row)
		#add the name if new class is found
		if(found[0] in gt.keys()):
			gt[found[0]].append(found[1:5])
		else:
			gt[found[0]] = [found[1:5]]

	#update TP and FP
	for p in predict:
		flag = 0
		for g in gt.get(p[0], []):
			if(IoU(p[1], g)>=0.5 and flag==0):
				mAP[p[0]]['TP'] += 1
				flag = 1
				break
		if(flag == 0):
			mAP[p[0]]['FP'] += 1

#main function
if __name__ == "__main__":
	#reading image names
	imlist = os.listdir(imgpath)
	#create mAP
	mAP = dict()
	file = open(classnames, 'r')
	classes = file.readlines()
	for c in classes:
		mAP[c[:-1]] = {'TP':0, 'FP':0}
	# tiny_yolo
	score = 0
	#load model
	net = dn.load_net(root+"/cfg/tiny-yolo-voc.cfg", root+"/tiny-yolo-voc.weights", 0)
	meta = dn.load_meta(root+"/cfg/voc_fp.data")
	#test
	tstart = time.time()
	for image in imlist:
		updatemAP(net, meta, image)
	tend = time.time()

	#mAP score 
	for c in mAP:
		if(mAP[c]['TP']+mAP[c]['FP'] != 0):
			score += mAP[c]['TP']/float(mAP[c]['TP']+mAP[c]['FP'])
	score = score/len(mAP)
	print("Tiny yolo Resulted mAP",score)
	print('Time taken is :',tend-tstart)
	result_file = open("result.txt", "w")
	result_file.write("Tiny yolo Resulted mAP score :"+str(score)+'\n\n')
	result_file.write("Time :" + str(tend-tstart)+'\n')
	result_file.close()
	# yolo
	mAP = dict()
	file = open(classnames, 'r')
	classes = file.readlines()
	for c in classes:
		mAP[c[:-1]] = {'TP':0, 'FP':0}
	# tiny_yolo
	score = 0
	net2 = dn.load_net(root+"/cfg/yolo-voc.cfg", root+"/yolo-voc.weights", 0)
	meta2 = dn.load_meta(root+"/cfg/voc_fp.data")

	#test
	tstart = time.time()
	for image in imlist:
		updatemAP(net2, meta2, image)
	tend = time.time()

	#mAP score 
	for c in mAP:
		if(mAP[c]['TP']+mAP[c]['FP'] != 0):
			score += mAP[c]['TP']/float(mAP[c]['TP']+mAP[c]['FP'])
	score = score/len(mAP)
	result_file = open("result2.txt", "w")
	print("Yolo Resulted mAP",score)
	print('Time taken is :',tend-tstart)

	result_file.write("Yolo Resulted mAP score :"+str(score)+'\n\n')
	result_file.write("Time :" + str(tend-tstart)+'\n')
	result_file.close()