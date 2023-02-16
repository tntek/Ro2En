
import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
# import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import sys
sys.path.append(os.getcwd())
from models.models import Encoder
# Environment Encoder


#N=number of environments; NP=Number of Paths
def load_dataset(N=100,NP=4000):

	Q = Encoder()
	Q.load_state_dict(torch.load('models/source/cae_encoder4.pkl'))
	print("models/source/cae_encoder4.pkl")
	if torch.cuda.is_available():
		Q.cuda()

		
	obs_rep=np.zeros((N,28),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile('data/3d/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(len(temp)//3,3)
		obstacles=np.zeros((1,6000),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[i]=output.numpy()



	
	## calculating length of the longest trajectory
	# 找到所有路径中点数最多的那条max
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	path_i = 0
	path_j = 0
	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
					path_i = i
					path_j = j
			
	# 存储N×NP条路进，默认路进长度为前面找到的max，不够用0补齐
	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					

	dataset=[]
	targets=[]
	for i in range(0,N):
		for j in range(0,NP):	# 第i个环境的第j条路进
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):	# 遍历这条路进的所以点
					data=np.zeros(34,dtype=np.float32)	# data是起始点，目标点，特征空间的拼接
					for k in range(0,28):	# 先存特征空间
						data[k]=obs_rep[i][k]
					# 存起始，目标点
					data[28]=paths[i][j][m][0]
					data[29]=paths[i][j][m][1]
					data[30]=paths[i][j][m][2]
					data[31]=paths[i][j][path_lengths[i][j]-1][0]
					data[32]=paths[i][j][path_lengths[i][j]-1][1]
					data[33]=paths[i][j][path_lengths[i][j]-1][2]
						
					targets.append(paths[i][j][m+1])
					dataset.append(data)
			
	data=list(zip(dataset,targets))
	random.shuffle(data)	
	dataset,targets=list(zip(*data))
	return 	np.asarray(dataset),np.asarray(targets) 

#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000):

	obc=np.zeros((N,10,3),dtype=np.float32)	# 后两个维度是障碍物个数，以及所处坐标空间维数
	temp=np.fromfile('data/3d/obs.dat')
	obs=temp.reshape(len(temp)//3,3)

	temp=np.fromfile('data/3d/obs_perm2.dat',np.int32)
	perm=temp.reshape(temp.size//10,10)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,10):
			for k in range(0,3):
				obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
	Q = Encoder()
	Q.load_state_dict(torch.load('models/source/cae_encoder3.pkl'))
	if torch.cuda.is_available():
		Q.cuda()
	
	obs_rep=np.zeros((N,28),dtype=np.float32)	
	k=0
	for i in range(s,s+N):
		temp=np.fromfile('data/3d/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(len(temp)//3,3)
		obstacles=np.zeros((1,6000),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[k]=output.numpy()
		k=k+1
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			

	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					



	return 	obc,obs_rep,paths,path_lengths
	
def load_3d_dataset(N=100,NP=4000):

	Q = Encoder()
	Q.load_state_dict(torch.load('models/source/3d/encoder/cae_offset03_encoder1.pkl'))
	if torch.cuda.is_available():
		Q.cuda()

		
	obs_rep=np.zeros((N,60),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloudsssss
		temp=np.fromfile('data/3d/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(len(temp)//3,3)
		obstacles=np.zeros((1,6000),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[i]=output.numpy()



	
	## calculating length of the longest trajectory
	# 找到所有路径中点数最多的那条max
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	path_i = 0
	path_j = 0
	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
					path_i = i
					path_j = j
			
	# 存储N×NP条路进，默认路进长度为前面找到的max，不够用0补齐
	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					

	dataset=[]
	targets=[]
	for i in range(0,N):
		for j in range(0,NP):	# 第i个环境的第j条路进
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):	# 遍历这条路进的所以点
					data=np.zeros(66,dtype=np.float32)	# data是起始点，目标点，特征空间的拼接
					for k in range(0,60):	# 先存特征空间
						data[k]=obs_rep[i][k]
					# 存起始，目标点
					data[60]=paths[i][j][m][0]
					data[61]=paths[i][j][m][1]
					data[62]=paths[i][j][m][2]
					data[63]=paths[i][j][path_lengths[i][j]-1][0]
					data[64]=paths[i][j][path_lengths[i][j]-1][1]
					data[65]=paths[i][j][path_lengths[i][j]-1][2]
						
					targets.append(paths[i][j][m+1])
					dataset.append(data)
			
	data=list(zip(dataset,targets))
	random.shuffle(data)	
	dataset,targets=list(zip(*data))
	return 	np.asarray(dataset),np.asarray(targets) 


def load_3d_test_dataset(ipath, N=100,NP=200, s=0,sp=4000):

	obc=np.zeros((N,10,3),dtype=np.float32)	# 后两个维度是障碍物个数，以及所处坐标空间维数
	temp=np.fromfile('data/3d/obs.dat')
	obs=temp.reshape(len(temp)//3,3)

	temp=np.fromfile('data/3d/obs_perm2.dat',np.int32)
	perm=temp.reshape(temp.size//10,10)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,10):
			for k in range(0,3):
				obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
	Q = Encoder()
	Q.load_state_dict(torch.load(ipath))
	if torch.cuda.is_available():
		Q.cuda()
	
	obs_rep=np.zeros((N,60),dtype=np.float32)	
	k=0
	for i in range(s,s+N):
		temp=np.fromfile('data/3d/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(len(temp)//3,3)
		obstacles=np.zeros((1,6000),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[k]=output.numpy()
		k=k+1
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			

	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='data/3d/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					



	return 	obc,obs_rep,paths,path_lengths