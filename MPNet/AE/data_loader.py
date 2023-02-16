import ssl
import numpy as np
# from pcl_visualizer import normalize_point_cloud



def load_dataset(N=30000):

    obstacles = np.zeros((N,6000),dtype=np.float32)
    for i in range(0,N):
        print("load env %d"%i)
        temp = np.fromfile('data/3d/obs_cloud/obc%d.dat'%i)
        temp = temp.reshape(len(temp)//3,3)
        obstacles[i] = temp.flatten()

    
    return obstacles
