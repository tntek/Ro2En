import argparse
import os
from sqlite3 import Timestamp
import sys
import math
import numpy as np

# from klog import prRed
sys.path.append(os.getcwd())
import torch
from torch import nn
from torch.autograd import Variable
from models.models import Encoder,Decoder
# from science_utils_k.utils.Logger import Logger
import time
# from klog import *
# cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
envs_obs_center = np.load("data/envs30000_obs_center.npy")
mse_loss = nn.MSELoss()
lam=1e-3
batch_count = 0





def load_dataset(N=30000):

    obstacles = np.zeros((N,6000),dtype=np.float32)
    for i in range(0,N):
        # temp = np.load('data/replace5/augmentation/obs/obs%d.npy'%i)
        
        temp = np.fromfile('data/3d/obs_cloud/obc%d.dat'%i)
        temp = temp.reshape(len(temp)//3,3)
        # print(np.mean(temp[:200,:],axis=0))
        # temp = normalize_point_cloud(temp)[0]
        obstacles[i] = temp.flatten()

    
    return obstacles


def caculate_obs_offset(decode_env, batchsize):
    obs_offset_loss = torch.tensor(0.0).cuda()
    source_env_centroid = torch.from_numpy(envs_obs_center[batch_count*batchsize:batch_count*batchsize+batchsize].astype(np.float32)).cuda()
    decode_env_centroid = torch.zeros((batchsize,10,3)).cuda()
    # for i in range(batchsize):
    #     for j in range(10):
    #         decode_env_centroid[i][j] = decode_env[i].reshape(2000,3)[j*200:(j+1)*200,:].mean(dim=0)
            
    # return mse_loss(source_env_centroid,decode_env_centroid)
    ####---
    for i in range(batchsize):
        for j in range(10):
            source_env_centroid = envs_obs_center[batch_count*batchsize+i][j]
            decode_env_centroid = decode_env[i].reshape(2000,3)[j*200:(j+1)*200,:].mean(dim=0)
            obs_offset = torch.from_numpy(source_env_centroid).cuda() - decode_env_centroid
            # obs_offset_loss += torch.Tensor.sqrt(obs_offset[0]**2 + obs_offset[1]**2 + obs_offset[2]**2)
            obs_offset_loss += obs_offset[0]**2 + obs_offset[1]**2 + obs_offset[2]**2
            
    
    
    return obs_offset_loss / batchsize 



def loss_function(W, x, recons_x, h):
    mse = mse_loss(recons_x, x)
    """
    W is shape of N_hidden x N. So, we do not need to transpose it as opposed to http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    """
    dh = h*(1-h) # N_batch x N_hidden
    contractive_loss = torch.sum(Variable(W)**2, dim=1).sum().mul_(lam)
    batchsize = x.shape[0]
    offset_loss = caculate_obs_offset(recons_x,batchsize)
    return  contractive_loss + mse + offset_loss*0.3


def main(args):	
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print("load data")
    obs = load_dataset()

    encoder = Encoder()
    # encoder.load_state_dict(torch.load("models/source/3d/encoder/cae_encoder2.pkl"))
    decoder = Decoder()
    # decoder.load_state_dict(torch.load("models/source/3d/encoder/cae_decoder2.pkl"))
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    
    params = list(encoder.parameters())+list(decoder.parameters())
    # params = list(encoder.parameters())
    optimizer = torch.optim.Adagrad(params)
    total_loss=[]
    for epoch in range(args.num_epochs):
        print("epoch" + str(epoch))
        avg_loss=0
        global batch_count
        batch_count = 0
        for i in range(0, len(obs), args.batch_size):
            decoder.zero_grad()
            encoder.zero_grad()
            if i+args.batch_size<len(obs):
                inp = obs[i:i+args.batch_size]
            else:
                inp = obs[i:]
            inp=torch.from_numpy(inp)
            inp =Variable(inp).cuda()
            # ===================forward=====================
            h = encoder(inp)
            output = decoder(h)
            # keys=list(encoder.state_dict().keys())
            W=encoder.state_dict()['encoder.6.weight'] # regularize or contracting last layer of encoder. Print keys to displace the layers name. 
            loss = loss_function(W,inp,output,h)
            
            avg_loss=avg_loss+loss.item()
            # ===================backward====================
            loss.backward()
            optimizer.step()
            batch_count += 1
        print("--average loss:")
        
        print(avg_loss/(len(obs)/args.batch_size))
        # total_loss.append(avg_loss/(len(obs)/args.batch_size))

    avg_loss=0
    for i in range(len(obs)-5000, len(obs), args.batch_size):
        inp = obs[i:i+args.batch_size]
        inp=torch.from_numpy(inp)
        inp =Variable(inp).cuda()
        # ===================forward=====================
        output = encoder(inp)
        output = decoder(output)
        loss = mse_loss(output,inp)
        avg_loss=avg_loss+loss.item()
        # ===================backward====================
    print("--Validation average loss:")
    print(avg_loss/(5000/args.batch_size))


    
    torch.save(encoder.state_dict(),os.path.join(args.model_path,'source/3d/encoder/cae_offset03_encoder1.pkl'))
    torch.save(decoder.state_dict(),os.path.join(args.model_path,'source/3d/encoder/cae_offset03_decoder1.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
    parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
    parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
    parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--input_size', type=int , default=18, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
