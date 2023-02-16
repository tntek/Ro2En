import random
# from turtle import circle
# import pcl_visualizer
import numpy as np
import math


def pc_normalize(pc):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    # 求质心，也就是一个平移量，实际上就是求均值
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m
    return pc


def generateCirclePointCloud(R,center):
    point_cloud = np.empty((200,3))
    for i in range(0, 200):
        # r = random.randint(0,R)
        r = R
        sin1  = random.uniform(-1,1)
        sin2 = random.uniform(-1,1)
        z = sin1 * r
        y = sin2 * math.sqrt(r**2-z**2)
        x = (1 if random.uniform(-1,1)>0 else -1 )* math.sqrt(r**2-z**2-y**2)
        
        point_cloud[i] = np.array(
            [
                center[0] + x,
                center[1] + y,
                center[2] + z
            ])
        
    return point_cloud

def generateSquarePointCloud(R,center):
    point_cloud = np.empty((200,3))
    for i in range(0, 200):
        r = R
        fix_flag = random.randint(0,2)
        if fix_flag == 0:
            # fix x 
            x = (1 if random.uniform(-1,1)>0 else -1 ) * r
            y = random.uniform(-r,r)
            z = random.uniform(-r,r)
        elif fix_flag == 1:
            #fix y
            y = (1 if random.uniform(-1,1)>0 else -1 ) * r
            x = random.uniform(-r,r)
            z = random.uniform(-r,r)
        elif fix_flag == 2:
            # fix z
            z = (1 if random.uniform(-1,1)>0 else -1 ) * r
            y = random.uniform(-r,r)
            x = random.uniform(-r,r)
            
        point_cloud[i] = np.array(
            [
                center[0] + x,
                center[1] + y,
                center[2] + z
            ])
        
    return point_cloud


def generateSquareBorder(center,r):
    
    borders = []
    for x in range(len(center)):
        border = np.array([
            [center[x][0]+r[x][0], center[x][1]-r[x][1],center[x][2]+r[x][2]],
            [center[x][0]+r[x][0], center[x][1]+r[x][1],center[x][2]+r[x][2]],
            [center[x][0]+r[x][0], center[x][1]-r[x][1],center[x][2]-r[x][2]],
            [center[x][0]+r[x][0], center[x][1]+r[x][1],center[x][2]-r[x][2]],
            [center[x][0]-r[x][0], center[x][1]-r[x][1],center[x][2]+r[x][2]],
            [center[x][0]-r[x][0], center[x][1]+r[x][1],center[x][2]+r[x][2]],
            [center[x][0]-r[x][0], center[x][1]-r[x][1],center[x][2]-r[x][2]],
            [center[x][0]-r[x][0], center[x][1]+r[x][1],center[x][2]-r[x][2]]
        ])
        borders.append(border.astype(np.float32))
    
    return borders
    
def generateEnv(square_num, circle_num):
    if (square_num + circle_num) > 10:
        print("obs_num should not over 10!")
        return
    
    point_cloud = []
    obs_center_set = []
    
    if square_num > 0:
        for i in range(square_num):
            obs_center = [
                random.uniform(-40,40),
                random.uniform(-40,40),
                random.uniform(-40,40),
            ]
            square_pc = generateSquarePointCloud(5, obs_center)
            point_cloud.append(square_pc)
            obs_center_set.append(obs_center)
    
    if circle_num > 0:
        for i in range(circle_num):
            obs_center = [
                random.uniform(-40,40),
                random.uniform(-40,40),
                random.uniform(-40,40),
            ]
            circle_pc = generateCirclePointCloud(8, obs_center)
            point_cloud.append(circle_pc)
            obs_center_set.append(obs_center)
    
    env = np.array(point_cloud).astype(np.float32)
    # np.random.shuffle(env)
    return env,np.array(obs_center_set).astype(np.float32)