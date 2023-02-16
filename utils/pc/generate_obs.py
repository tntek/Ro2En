import random
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

def get_centroid(pc):
    """得到某个物体的中心点

    Args:
        pc (array): 某个物体的点云数据，numpy格式

    Returns:
        array: 中心点
    """

    return np.mean(pc, axis=0)

def generateCirclePointCloud(R, center, num=200):
    """生成一个圆的表面采样的点

    Args:
        R (float): 圆的半径
        center (list): 圆的中心位置
        num (int): 采样点个数

    Returns:
        array: 返回一个圆的表面采样的点云数据
    """

    point_cloud = np.empty((num,3))
    r = R

    for i in range(0, num):
        # r = random.randint(0,R)
        
        sin1  = random.uniform(-1,1)    # 仰角
        sin2 = random.uniform(-1,1) # 平面角
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

def generateSquarePointCloud(R, center, num=200):
    """生成一个正方体表面采样的点云数据

    Args:
        R (float): 正方体的半径，即0.5正方体的直径
        center (list): 正方体的中心点
        num (int): 采样点个数

    Returns:
        array: 返回一个正方体表面采样的点云数据
    """

    point_cloud = np.empty((num,3))
    r = R

    for i in range(0, num):
        
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

def generateCuboidPointCloud(edge, center, num=200):
    """生成一个长方体表面采样的点云数据

    Args:
        edge (list): 长方体的长宽高
        center (float): 长方体的中心点

    Returns:
        array: 返回一个正方体表面采样的点云数据
    """

    point_cloud = np.empty((num,3))
    r = 0.5 * np.array(edge)

    for i in range(0, num):
        
        fix_flag = random.randint(0,2)
        if fix_flag == 0:
            # fix x 
            x = (1 if random.uniform(-1,1)>0 else -1 ) * r[0]
            y = random.uniform(-r[1],r[1])
            z = random.uniform(-r[2],r[2])
        elif fix_flag == 1:
            #fix y
            y = (1 if random.uniform(-1,1)>0 else -1 ) * r[1]
            x = random.uniform(-r[0],r[0])
            z = random.uniform(-r[2],r[2])
        elif fix_flag == 2:
            # fix z
            z = (1 if random.uniform(-1,1)>0 else -1 ) * r[2]
            y = random.uniform(-r[1],r[1])
            x = random.uniform(-r[0],r[0])
            
        point_cloud[i] = np.array(
            [
                center[0] + x,
                center[1] + y,
                center[2] + z
            ])

    return point_cloud

def generateCylinderPointCloud(R, H, center, num=200):
    """生成一个圆柱体表面采样的点云数据

    Args:
        R (float): 半径
        H (float): 高度
        center (list): 圆柱体中心点
        num (int): 采样点个数

    Returns:
        array: 返回一个圆柱体表面采样的点云数据
    """
    point_cloud = np.empty((num,3))
    r = R

    for i in range(num):
        draw_flag = random.randint(0,9)
        sin1 = random.uniform(-1,1) # 平面角

        # 画侧面
        if draw_flag < 8:
            y = sin1 * r
            x =  (1 if random.uniform(-1,1)>0 else -1 ) * math.sqrt(r**2 - y**2)
            point = np.array([
                center[0] + x,
                center[1] + y,
                center[2] + random.uniform(-1,1)*H/2
            ])
        # 画上下面
        else:
            bottom_r = random.uniform(0, r)
            y = sin1 * bottom_r
            x =  (1 if random.uniform(-1,1)>0 else -1 ) * math.sqrt(bottom_r**2 - y**2)
            point = np.array([
                center[0] + x,
                center[1] + y,
                center[2] + (1 if random.uniform(-1,1)>0 else -1 ) * H/2
            ])
        point_cloud[i] =point

    return point_cloud

def generateConePointCloud(R, H, center, num=200):
    """生成一个圆锥体表面采样的点云数据

    Args:
        R (float): 底面半径
        H (float): 高度
        center (list): 圆锥体底面中心点
        num (int, optional): 点云个数. Defaults to 200.

    Returns:
        array: 返回一个圆锥体表面采样的点云数据
    """

    point_cloud = np.empty((num,3))
    r = R

    for i in range(num):
        draw_flag = random.randint(0,9)
        sin1 = random.uniform(-1,1)
        
        # 画底面
        if draw_flag > 7: 
            bottom_r = random.uniform(0, r)
            y = sin1 * bottom_r
            x =  (1 if random.uniform(-1,1)>0 else -1 ) * math.sqrt(bottom_r**2 - y**2)
            point = np.array([
                center[0] + x,
                center[1] + y,
                center[2]
            ])
        else:
            bottom_r = random.uniform(0, r)
            y = sin1 * bottom_r
            x =  (1 if random.uniform(-1,1)>0 else -1 ) * math.sqrt(bottom_r**2 - y**2)
            point = np.array([
                center[0] + x,
                center[1] + y,
                center[2] + H*(1-bottom_r/r)
            ])
        
        point_cloud[i] = point

    return point_cloud


def generateSquareBorder(center,r):
    """生成正方体的顶点

    Args:
        center (list): 正方体的中心
        r (float): 正方体的半径

    Returns:
        list: 返回一个正方体顶点数组
    """

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
