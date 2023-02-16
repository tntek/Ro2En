import numpy as np
import random
from utils.pc.generate_obs import *
from utils.pc.visulizer import drawPointCloud

box_size = 20

def IsInCollision(x, obs_center, obs_collision):
    s=np.zeros(3,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    s[2] = x[2]
    box_collision = box_size+10

    for i in range(0,10):
        cf=False
        if abs(obs_center[i][0] - s[0]) < obs_collision[i][0] and abs(obs_center[i][1] - s[1]) < obs_collision[i][1] and abs(obs_center[i][2] - s[2]) < obs_collision[i][2]:
            # print("Have collsion!!!!")
            return True
    return False

def steerTo (start, end, obs_center, obs_collision):

    DISCRETIZATION_STEP=0.01
    dists=np.zeros(3,dtype=np.float32)
    for i in range(0,3): 
        dists[i] = end[i] - start[i]

    distTotal = 0.0
    for i in range(0,3): 
        distTotal =distTotal+ dists[i]*dists[i]

    distTotal = math.sqrt(distTotal)
    if distTotal>0:
        incrementTotal = distTotal/DISCRETIZATION_STEP
        for i in range(0,3): 
            dists[i] =dists[i]/incrementTotal


        numSegments = int(math.floor(incrementTotal))

        stateCurr = np.zeros(3,dtype=np.float32)
        for i in range(0,3): 
            stateCurr[i] = start[i]
        for i in range(0,numSegments):

            if IsInCollision(stateCurr, obs_center, obs_collision):
                return 0

            for j in range(0,3):
                stateCurr[j] = stateCurr[j]+dists[j]


        if IsInCollision(end,obs_center, obs_collision):
            return 0


    return 1

def generate_env():
    env = np.empty((30000,2000,3))
    center = np.empty((30000,10,3))
    obs_collision = np.empty((30000,10,3))
    
    for i in range(30000):
        
        for j in range(10):
            flag = random.randint(0,9)
            obs_center = [
                random.uniform(-box_size,box_size),
                random.uniform(-box_size,box_size),
                random.uniform(-box_size,box_size),
            ]
            center[i][j] = np.array(obs_center)
            if flag in (0,1):   # 生成圆
                r = random.uniform(4,5)
                obs = generateCirclePointCloud(r,
                    center=obs_center
                    )
                obs_collision[i][j] = np.array([r,r,r])
            elif flag in (2,3):     # 生成圆锥体
                r = random.uniform(4,5)
                h = random.uniform(8,10)
                obs = generateConePointCloud(r,h,center=obs_center)
                center[i][j][2] = center[i][j][2] + h/2.0
                obs_collision[i][j] = np.array([r,r,h/2.0])
            elif flag in (4,5):     # 生成圆柱
                r = random.uniform(4,5)
                h = random.uniform(8,10)
                obs = generateCylinderPointCloud(r,h,center=obs_center)
                obs_collision[i][j] = np.array([r,r,h/2.0])
            elif flag in (6,7):
                r = random.uniform(4,5)     # 生成正方形
                obs = generateSquarePointCloud(r,center=obs_center)
                obs_collision[i][j] = np.array([r,r,r])
            else:
                edge = [
                    random.uniform(5,10),
                    random.uniform(5,10),
                    random.uniform(5,10),
                ]
                obs = generateCuboidPointCloud(edge=edge, center=obs_center)
                obs_collision[i][j] = (np.array(edge)/2.0)
            
            env[i][j*200:(j+1)*200,:] = obs
        np.save("data/complex_3d/augmentation/obs/obs%d.npy"%i,env[i])
        print("generate data/complex_3d/augmentation/obs/obs%d.npy success!!!"%i)

    np.save("data/complex_3d/augmentation/complex_3d_augmentation_obs.npy",env)
    np.save("data/complex_3d/augmentation/complex_3d_augmentation_obs_centers.npy",center)
    np.save("data/complex_3d/augmentation/complex_3d_augmentation_obs_collision.npy",obs_collision)

def initstate():
    obs_center = np.load("data/replace3/obs_center.npy")
    obc = []
    envs = np.load("data/replace3/obs.npy")
    obs_collision = np.load("data/replace3/obs_collision.npy")
    init_state_list = np.empty((100,10,2,3))
    for i in range(100):
        obc = obs_center[i]
        for j in range(10):
            r = 4.5
            start_near_obc = random.randint(0,9)
            distance = random.uniform(1,4)
            start_sin1  = random.uniform(-1,1)
            start_sin2 = random.uniform(-1,1)
            start_z = start_sin1 * (r+distance)
            start_y = start_sin2 * math.sqrt((r+distance)**2-start_z**2)
            start_x = (1 if random.uniform(-1,1)>0 else -1 )*  math.sqrt((r+distance)**2-start_z**2-start_y**2)
            start=np.array([
                obc[start_near_obc][0]+start_x,
                obc[start_near_obc][1]+start_y,
                obc[start_near_obc][2]+start_z,
                # 1,
                # 0,
                # 0
            ]).astype(np.float32)
            while IsInCollision(start,obc,obs_collision[i]):
                print("start happenning collison")
                start_sin1  = random.uniform(-1,1)
                start_sin2 = random.uniform(-1,1)
                distance = random.uniform(1,4)
                start_z = start_sin1 * (r+distance)
                start_y = start_sin2 * math.sqrt((r+distance)**2-start_z**2)
                start_x = (1 if random.uniform(-1,1)>0 else -1 )*  math.sqrt((r+distance)**2-start_z**2-start_y**2)
                start=np.array([
                    obc[start_near_obc][0]+start_x,
                    obc[start_near_obc][1]+start_y,
                    obc[start_near_obc][2]+start_z,
                    # 1,
                    # 0,
                    # 0
                ]).astype(np.float32)

            goal_near_obc = random.randint(0,9)
            goal_sin1  = random.uniform(-1,1)
            goal_sin2 = random.uniform(-1,1)
            distance = random.uniform(1,4)
            goal_z = goal_sin1 * (r+distance)
            goal_y = goal_sin2 * math.sqrt((r+distance)**2-goal_z**2)
            goal_x = (1 if random.uniform(-1,1)>0 else -1 )*  math.sqrt((r+distance)**2-goal_z**2-goal_y**2)
            goal=np.array([
                obc[goal_near_obc][0]+goal_x,
                obc[goal_near_obc][1]+goal_y,
                obc[goal_near_obc][2]+goal_z,
                # 1,
                # 0,
                # 0
            ]).astype(np.float32)
            
            while IsInCollision(goal,obc,obs_collision[i]) or steerTo(start, goal,obc, obs_collision[i]):
                print("goal happenning collison")
                goal_sin1  = random.uniform(-1,1)
                goal_sin2 = random.uniform(-1,1)
                distance = random.uniform(1,4)
                goal_z = goal_sin1 * (r+distance)
                goal_y = goal_sin2 * math.sqrt((r+distance)**2-goal_z**2)
                goal_x = (1 if random.uniform(-1,1)>0 else -1 )*  math.sqrt((r+distance)**2-goal_z**2-goal_y**2)
                goal=np.array([
                    obc[goal_near_obc][0]+goal_x,
                    obc[goal_near_obc][1]+goal_y,
                    obc[goal_near_obc][2]+goal_z,
                    # 1,
                    # 0,
                    # 0
                ]).astype(np.float32)
            init_state_list[i][j][0] = start
            init_state_list[i][j][1] = goal
            init_state = np.empty((2,6))
            init_state[0][:3] = start
            init_state[0][3:] = [1,0,0]
            init_state[1][:3] = goal
            init_state[1][3:] = [1,0,0]
            # drawPointCloud([envs[i],init_state])
            print("obs%d,initstate%d generate success!!!" % (i,j))
    np.save("data/replace3/initstate.npy", init_state_list)
    # ---generate init and target state----

# initstate()