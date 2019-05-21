import matplotlib.pyplot as plt
import matplotlib.image as im
from scipy.misc import imread, imsave, imresize
from scipy import ndimage, io
from scipy.ndimage import gaussian_filter
from scipy import signal as sig
from matplotlib.patches import Circle
from numpy import linalg as LA
import numpy as np
import pickle
import cv2
#PART1
def filter_responses():
	filters=io.loadmat('filters.mat')
	imgs=[[] for i in range(6)]
	imgs[0]=imread('cardinal1.jpg',flatten=True)
	imgs[1]=imread('cardinal2.jpg',flatten=True)
	imgs[2]=imread('leopard1.jpg',flatten=True)
	imgs[3]=imread('leopard2.jpg',flatten=True)
	imgs[4]=imread('panda1.jpg',flatten=True)
	imgs[5]=imread('panda2.jpg',flatten=True)
	for i in range(len(imgs)):
		imgs[i]=imresize(imgs[i],(100,100))
	fig, axis = plt.subplots(nrows=2, ncols=4)
	#convolving and subplotting
	for j in range(2,len(filters['F'])):
		axis[0][0].imshow(filters['F'][:,:,j])
		axis[0][0].aspect="auto"
		axis[0][1].axis('off')
		axis[0][2].imshow(ndimage.convolve(imgs[0], filters['F'][:,:,j]))
		axis[0][2].set_title("cardinal1")
		axis[0][3].imshow(ndimage.convolve(imgs[1], filters['F'][:,:,j]))
		axis[0][3].set_title("cardinal2")
		axis[1][0].imshow(ndimage.convolve(imgs[2], filters['F'][:,:,j]))
		axis[1][0].set_title("leopard1")
		axis[1][1].imshow(ndimage.convolve(imgs[3], filters['F'][:,:,j]))
		axis[1][1].set_title("leopard2")
		axis[1][2].imshow(ndimage.convolve(imgs[4], filters['F'][:,:,j]))
		axis[1][2].set_title("panda1")
		axis[1][3].imshow(ndimage.convolve(imgs[5], filters['F'][:,:,j]))
		axis[1][3].set_title("panda2")
		plt.savefig('testplot'+str(j)+'.png')
		#plt.show() 

#PART2
def computeTextureReprs(image,F):
	num_filters=len(F[0][0])
	height=len(image)
	width=len(image[0])
	texture_repr_concat=[]
	texture_repr_mean=[]
	responses=np.zeros((num_filters,height,width))
	#grayscale if 3D, else leave as it is
	if(image.ndim==3):
		imggs=np.dot(image,[0.299,0.587,.114])
	else:
		imggs=image
	#convolve filters with grayscale image
	for j in range(num_filters):
		responses[j]=ndimage.convolve(imggs, F[:,:,j])
	# calculate texture_repr_concat
	for d in range(num_filters):
		for i in range(height):
			for j in range(width):
				texture_repr_concat.append(responses[d][i][j])
	# calculate texture_repr_mean
	for i in range(num_filters):
		texture_repr_mean.append(np.sum(responses[i][:,:])/width/height)
	print(len(texture_repr_mean))		
	return [np.array(texture_repr_concat),np.array(texture_repr_mean)]

#PART3
def hybrid_images():
	imgs=[[],[]]
	im1=imread('baby_happy.jpg')
	im2=imread('baby_weird.jpg')
	im1=imresize(im1,(512,512))
	im2=imresize(im2,(512,512))
	im1=np.dot(im1,[0.299,0.587,.114])
	im2=np.dot(im2,[0.299,0.587,.114])
	im1_blur=gaussian_filter(im1,sigma=2)
	im2_blur=gaussian_filter(im2,sigma=2)
	im2_detail=im2-im2_blur
	hybrid=im1_blur+im2_detail
	imsave('hybrid.jpg', hybrid)

#PART4
def plotting(R,axs,axis,img):

	count=0	
	corners=[]
	xes=[]
	yes=[]
	scores=[]
	for i in range(len(R)):
		for j in range(len(R[i])):
			if R[i][j]>0:
				corners.append((i,j))
				xes.append(j)
				yes.append(i)
				scores.append(R[i][j])
				circ = Circle((j,i),1,color='red')
				axis[axs].add_patch(circ)
				#print(R[i][j])
				count=count+1
	return [yes,xes,scores] 		
def extract_keypoints(img):
	k=0.05
	window_size=5
	if(img.ndim==3):
		imggs=np.dot(img,[0.299,0.587,.114])
	else:
		imggs=img
	Ix=cv2.Sobel(imggs, cv2.CV_64F, 1, 0)
	Iy = cv2.Sobel(imggs, cv2.CV_64F, 0, 1)
	#Iy,Ix=np.gradient(imggs)
	
	width=len(imggs[0])
	height=len(imggs)
	R=np.zeros((height,width))
	Ixx = Ix**2
	Ixy = Iy*Ix
	Iyy = Iy**2
	offset=int(window_size/2)
	for y in range(offset,height-offset):
		for x in range(offset,width-offset):
			M1=np.sum(Ixx[y-offset:y+offset+1, x-offset:x+offset+1])
			M2=np.sum(Ixy[y-offset:y+offset+1, x-offset:x+offset+1])
			M4=np.sum(Iyy[y-offset:y+offset+1, x-offset:x+offset+1])
			M=np.array([[M1, M2],[M2, M4]])
			b,v=LA.eig(M)
			determinant=b[0]*b[1]
			trace=b[0]+b[1]
			R[y][x]=np.linalg.det(M)-k*(np.trace(M)**2)
			if y<2 or y >height-3 or x<2 or x >width-3:
				R[y][x]=0
	#plotting image before thresholding supression
	
	fig, axis = plt.subplots(nrows=1, ncols=3)
	axis[0].imshow(img)
	axis[0].set_title("before_thresholding")
	plotting(R,0,axis,img)
	#non-maximum-supression
	threshold=np.percentile(R,99)
	R[R<threshold]=0
	#plotting image before non maximum supression
	axis[1].imshow(img)
	axis[1].set_title("before_supression")
	plotting(R,1,axis,img)
	#non-maximum-supression
	for i in range(1,len(R)-1):
		for j in range(1,len(R[i])-1):
			mx=np.max([R[i-1][j-1], R[i-1][j], R[i-1][j+1], R[i+1][j-1], R[i+1][j], R[i+1][j+1], R[i][j-1], R[i][j+1]])
			if (mx>R[i][j]):
				R[i][j]=0
	#plotting image after non maimum supression
	yes,xes,scores=plotting(R,2,axis,img)
	axis[2].imshow(img)
	axis[2].set_title("after_supression")
	plt.savefig('vis3.png')
	return {'yes':yes,'xes':xes,'scores':scores,'Ix': Ix,'Iy':Iy}

def compute_features(img,yes,xes,scores,Ix,Iy ):
	corners=np.concatenate((np.transpose([yes]), np.transpose([xes])),axis=1)
	lenx=len(Ix[0])
	leny=len(Ix)
	n=len(corners)
	feature=np.zeros((n,8))
	ncor=[] #ncor are the new corners(except border 5) where features will be calculated
	if(img.ndim==3):
		imggs=np.dot(img,[0.299,0.587,.114])
	else:
		imggs=img
	for (i,j) in corners:
		if (i<5) or (i >leny-6) or (j<5) or (j>lenx-6):
			pass
		else:
			ncor.append((i,j))
	ncor=np.array(ncor)
	mx=np.zeros((leny,lenx))
	thetha=np.zeros((leny,lenx))
	for i in range(1,leny-1):
		for j in range(1,lenx-1):
			Dr=max(imggs[i+1][j]-imggs[i-1][j],0.000001)# to avoid devision by 0
			Nr=imggs[i][j+1]-imggs[i][j-1]
			mx[i][j]=np.sqrt((Nr)**2+(Dr)**2)
			thetha[i][j]=np.arctan(Nr/Dr)
	print(thetha.shape)
	# Lx=Ix*2
	# Ly=Ly*2
	# mx=np.sqrt(Lx*Lx+Ly*Ly)
	# Ix[Ix==0]=0.0000001
	# thetha=np.arctan(Iy/Ix)
	minthetha=np.arctan(-np.Inf)
	maxthetha=np.arctan(np.Inf)
	hight=len(Ix)
	width=len(Ix[0])
	offset2=5
	n2=len(ncor)
	print(max(xes),max(yes))
	for i in range(n2):
		for y in range(-offset2,offset2+1):
			for x in range(-offset2,offset2+1):
				#print(yes[i],xes[i])
				bn=int((thetha[ncor[i][0]+y][ncor[i][1]+x]-minthetha)/(maxthetha-minthetha)*8)    #bn is the bin number
				if(bn==8):
					bn=7
				feature[i][bn]+=mx[ncor[i][0]+y][ncor[i][1]+x]
		feature[i] = feature[i] / np.linalg.norm(feature[i])
		feature[i]=np.minimum(feature[i],0.2)
		feature[i] = feature[i] / np.linalg.norm(feature[i])
	print(feature,feature.shape)
	return feature

def computeBOWRepr( features, means):

	k=len(means)
	n=len(features)
	means=np.array(means)
	features=np.array(features)
	clas=np.zeros(n)
	bow=np.zeros(k)
	#print(bow)
	#print(features,features.shape,means,means.shape)
	for i in range(n):
		diff=np.linalg.norm(np.subtract(features[i],means), axis=1)
		clas[i]=np.argmin(diff)
		bow[int(clas[i])]+=1
	normalizing_factor=np.sum(bow)
	bow_normalized=bow/normalizing_factor
	print ("BOW",bow)
	print (bow_normalized)
	return bow_normalized

def compareDiscriptions():
	imgs=[[] for i in range(6)]
	keypoints=[[] for i in range(6)]
	features=[[] for i in range(6)]
	bow=[[] for i in range(6)]
	texture_repr_concat=[[] for i in range(6)]
	texture_repr_concat=[[] for i in range(6)]
	bow_dist=np.zeros((6,6))
	tconcat_dist=np.zeros((6,6))
	tmean_dist=np.zeros((6,6))
	filters=io.loadmat('filters.mat')
	#print(filters['F'].shape)
	texture_repr_mean=[[] for i in range(6)]
	texture_repr_concat=[[] for i in range(6)]
	imgs[0]=imread('cardinal1.jpg',flatten=True)
	imgs[1]=imread('cardinal2.jpg',flatten=True)
	imgs[2]=imread('leopard1.jpg',flatten=True)
	imgs[3]=imread('leopard2.jpg',flatten=True)
	imgs[4]=imread('panda1.jpg',flatten=True)
	imgs[5]=imread('panda2.jpg',flatten=True)
	means=io.loadmat('means.mat')
	means=means['means']
	for i in range(len(imgs)):
		imgs[i]=imresize(imgs[i],(100,100)) 
		keypoints[i]=extract_keypoints(imgs[i])
		yes=keypoints[i]['yes']
		xes=keypoints[i]['xes']
		scores=keypoints[i]['scores']
		Ix=keypoints[i]['Ix']
		Iy=keypoints[i]['Iy']
		features[i]=compute_features(imgs[i],yes,xes,scores,Ix,Iy)
		bow[i]=computeBOWRepr(features[i],means)
		texture_repr_concat[i],texture_repr_mean[i]= computeTextureReprs(imgs[i],filters['F'])
	f = open('bow.pckl', 'wb')
	pickle.dump(bow, f)
	f.close()
	f = open('texture_repr_concat.pckl', 'wb')
	pickle.dump(texture_repr_concat, f)
	f.close()
	f = open('texture_repr_mean.pckl', 'wb')
	pickle.dump(texture_repr_mean, f)
	f.close()

	f = open('bow.pckl', 'rb')
	bow = pickle.load(f)
	f.close()
	f = open('texture_repr_concat.pckl', 'rb')
	texture_repr_concat = pickle.load(f)
	f.close()
	f = open('texture_repr_mean.pckl', 'rb')
	texture_repr_mean = pickle.load(f)
	f.close()

	for i in range(len(imgs)):
		for j in range(len(imgs)):
			bow_dist[i][j]=np.linalg.norm(bow[i]-bow[j])
			tconcat_dist[i][j]=np.linalg.norm(np.subtract(texture_repr_concat[i],texture_repr_concat[j]))
			tmean_dist[i][j]=np.linalg.norm(texture_repr_mean[i]-texture_repr_mean[j])
	avg_bow_in_class_dist=[bow_dist[0][1],bow_dist[2][3],bow_dist[4][5]]
	avg_tconcat_in_class_dist=[tconcat_dist[0][1],tconcat_dist[2][3],tconcat_dist[4][5]]
	avg_tmean_in_class_dist=[tmean_dist[0][1],tmean_dist[2][3],tmean_dist[4][5]]
	avg_bow_bw_class_dist=[i for i in bow_dist[0:1,2:5].tolist()]+[j for j in bow_dist[2:3,1:2].tolist()]+[k for k in bow_dist[2:3,4:5].tolist()]+[l for l in bow_dist[4:5,1:3].tolist()]
	avg_bow_bw_class_dist=sum(avg_bow_bw_class_dist,[])
	avg_tconcat_bw_class_dist=[i for i in tconcat_dist[0:1,2:5].tolist()]+[j for j in tconcat_dist[2:3,1:2].tolist()]+[k for k in tconcat_dist[2:3,4:5].tolist()]+[l for l in tconcat_dist[4:5,1:3].tolist()]
	avg_tconcat_bw_class_dist=sum(avg_tconcat_bw_class_dist,[])
	avg_tmean_bw_class_dist=[i for i in tmean_dist[0:1,2:5].tolist()]+[j for j in tmean_dist[2:3,1:2].tolist()]+[k for k in tmean_dist[2:3,4:5].tolist()]+[l for l in tmean_dist[4:5,1:3].tolist()]
	avg_tmean_bw_class_dist=sum(avg_tmean_bw_class_dist,[])
	avg_bow_dist=sum(avg_bow_in_class_dist)/sum(avg_bow_bw_class_dist)
	avg_tconcat_dist=sum(avg_tconcat_in_class_dist)/sum(avg_tconcat_bw_class_dist)
	avg_tmean_dist=sum(avg_tmean_in_class_dist)/sum(avg_tmean_bw_class_dist)
	print("avg_bow_ratio:",avg_bow_dist,"avg_tmean_ratio:",avg_tmean_dist,"avg_tconcat_ratio:",avg_tconcat_dist)

if __name__ == '__main__':
	#Part1
	# filter_responses()

	#Part2
	# image=imread('cardinal1.jpg')
	# filtr=io.loadmat('filters.mat')
	# filtrf=filtr["F"]
	# computeTextureReprs(image,filtrf)

	#Part3
	#hybrid_images()

	#Part4

	# img=imread('baby_happy.jpg')
	# keypoints=extract_keypoints(img)
	# f = open('keypoints.pckl', 'wb')
	# pickle.dump(keypoints, f)
	# f.close()'
	# f = open('img.pckl', 'wb')
	# pickle.dump(img, f)
	# f.close()

	#Part5
	# f = open('img.pckl', 'rb')
	# img = pickle.load(f)
	# f.close()
	# f = open('keypoints.pckl', 'rb')
	# keypoints = pickle.load(f)
	# f.close()
	# yes=keypoints['yes']
	# xes=keypoints['xes']
	# scores=keypoints['scores']
	# Ix=keypoints['Ix']
	# Iy=keypoints['Iy']
	# features=compute_features(img,yes,xes,scores,Ix,Iy)
	# f = open('features.pckl', 'wb')
	# pickle.dump(features, f)
	# f.close()

	#Part6
	# f = open('features.pckl', 'rb')
	# features = pickle.load(f)
	# f.close()

	# means=io.loadmat('means.mat')
	# means=means['means']
	# bow=computeBOWRepr(features,means)
	# f = open('bow.pckl', 'wb')
	# pickle.dump(bow, f)
	# f.close()

	#Part7

	compareDiscriptions()
