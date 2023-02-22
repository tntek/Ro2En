
from models.models import Encoder
import torch
#from generate_obs import  generateSquareBorder
#from pcl_visualizer import drawPath
import time,math
import numpy as np
from MPNet.model import MLP 
# from klog import *
import time
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))

obs_num = 10	#障碍物个数，2d下是7,3d下是10
TIMEOUT=2
box_size = 20

# Load trained model for path generation
mlp_path = "models/source/3d/mlp/cae_encoder_mlp_100_4000_PReLU_ae_dd_final.pkl"
encoder_path = "models/source/3d/encoder/cae_encoder.pkl"
env_path = "data/replace10/"
mlp = MLP(66, 3) # simple @D
# rp = relative_position()
# rp.load_state_dict(torch.load("models/source/3d/position/rp1.pkl"))
# mlp.load_state_dict(torch.load(mlp_path))
encoder = Encoder()
# encoder.load_state_dict(torch.load(encoder_path))
envs = np.load(env_path+"obs.npy").astype(np.float32)
obs_center = np.load(env_path+"obs_center.npy").astype(np.float32)
obs_collision = np.load(env_path+"obs_collision.npy").astype(np.float32)
init_state = np.load(env_path+"initstate.npy").astype(np.float32)
current_env = []
current_obc = []
current_collision = []

def IsInCollision(x):
    s=np.zeros(3,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    s[2] = x[2]
    box_collision = box_size+10
    if abs(s[0]) > box_collision or abs(s[1]) > box_collision or abs(s[2]) > box_collision:
        return True


    for i in range(0,10):
        if abs(current_obc[i][0] - s[0]) < current_collision[i][0] and abs(current_obc[i][1] - s[1]) < current_collision[i][1] and abs(current_obc[i][2] - s[2]) < current_collision[i][2]:
            # print("Have collsion!!!!")
            return True
    return False


def steerTo (start, end):

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

            if IsInCollision(stateCurr):
                return 0

            for j in range(0,3):
                stateCurr[j] = stateCurr[j]+dists[j]


        if IsInCollision(end):
            return 0


    return 1

# checks the feasibility of entire path including the path edges
def feasibility_check(path):

    for i in range(0,len(path)-1):
        ind=steerTo(path[i],path[i+1])
        if ind==0:
            return 0
    return 1


# checks the feasibility of path nodes only
def collision_check(path):

    for i in range(0,len(path)):
        if IsInCollision(path[i]):
            return 0
    return 1




#lazy vertex contraction 
def lvc(path):

    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(path[i],path[j])
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])

                return lvc(pc)
                
    return path


# Replanning
def replan_path(p,g,obs):
    step=0
    path=[]
    path.append(p[0])
    for i in range(1,len(p)-1):
        if not IsInCollision(p[i]):
            path.append(p[i])
    path.append(g)			
    new_path=[]
    for i in range(0,len(path)-1):
        target_reached=False 
        st=path[i]
        gl=path[i+1]
        steer=steerTo(st, gl)
        if steer==1:
            new_path.append(st)
            new_path.append(gl)
        else:
            itr=0
            pA=[]
            pA.append(st)
            pB=[]
            pB.append(gl)
            target_reached=0
            tree=0
            while target_reached==0 and itr<50 :
                itr=itr+1
                if tree==0:
                    ip1=torch.cat((obs,st,gl))
                    st=mlp(ip1)
                    st=st.data.cpu()
                    # print("generate a node on treeA:" + str(st))
                    print("generate a node on treeA:" + str(st))
                    pA.append(st)
                    tree=1
                else:
                    ip2=torch.cat((obs,gl,st))
                    gl=mlp(ip2)
                    gl=gl.data.cpu()
                    # print("generate a node on treeB:" + str(gl))
                    print("generate a node on treeB:" + str(gl))
                    pB.append(gl)
                    tree=0		
                target_reached=steerTo(st, gl)
            if target_reached==0:
                return 0
            else:
                for p1 in range(0,len(pA)):
                    new_path.append(pA[p1])
                for p2 in range(len(pB)-1,-1,-1):
                    new_path.append(pB[p2])

    return new_path	
    
def generatePath():
    tp=0    # 总共路径
    fp=0 # 可行路径
    tot=[]
    et=[]
    
    
    for i in range(100):
        global current_env,current_obc,current_collision
        current_env = envs[i]
        current_obc = obs_center[i]
        current_collision = obs_collision[i]
        for j in range(0,10):
            start = init_state[i][j][0]
            goal = init_state[i][j][1]
            # print("use complex3d,complex_augment_cae_encoder.pkl")
            # print("use cae_augment_noOffset_encoder")
            obs = encoder(torch.from_numpy(current_env.ravel()))
            # obs_pos = rp(torch.from_numpy(current_env.ravel()))
            # obs=torch.cat((obs,obs_pos))
            #start and goal for bidirectional generation
            ## starting point
            start1=torch.from_numpy(start)
            goal2=torch.from_numpy(start)
            ##goal point
            goal1=torch.from_numpy(goal)
            start2=torch.from_numpy(goal)
            ##obstacles
            # obs=torch.from_numpy
            # (obs)
            ##generated paths
            path1=[] 
            path1.append(start1)
            path2=[]
            path2.append(start2)
            target_reached=0
            step=0	
            path=[] # stores end2end path by concatenating path1 and path2
            tree=0	
            tic = time.clock()
            
            # 迭代搜索路经target_reached为标志位，判断现在的节点是否达到路径，算法会搜索80次
            tp=tp+1
            while target_reached==0 and step<80 :
                toc = time.clock()
                if (toc - tic) > TIMEOUT:
                    target_reached=0
                    prRed("TIMEOUT!!!tic :%f ,toc:%f"%(tic,toc))
                    break
                # print("Start Planning %d.th Path, step %d ..." %(j, step))
                print("Start Planning %d.th Path, step %d on enviroment %d..." %(j, step, i))
                step=step+1
                
                # 第一棵生成树
                if tree==0:
                    inp1=torch.cat((obs,start1,start2))
                    start1=mlp(inp1)	# 预测节点
                    start1=start1.data.cpu()
                    # print("generate a node on tree1:" + str(start1))
                    print("generate a node on tree1:" + str(start1))
                    path1.append(start1)
                    tree=1	# 换方向
                # 第二颗生成树
                else:
                    inp2=torch.cat((obs,start2,start1))
                    start2=mlp(inp2)
                    start2=start2.data.cpu()
                    # print("generate a node on tree2:" + str(start2))
                    print("generate a node on tree2:" + str(start2))
                    path2.append(start2)
                    tree=0
                target_reached=steerTo(start1,start2) # 判断两颗树的末端是否连接
            

            if target_reached==1:
                for p1 in range(0,len(path1)):
                    path.append(path1[p1])
                for p2 in range(len(path2)-1,-1,-1):
                    path.append(path2[p2])
                                                
                
                path=lvc(path)	# 去冗余
                indicator=feasibility_check(path)
                if indicator==1:
                    toc = time.clock()
                    t=toc-tic
                    if t > TIMEOUT:
                        indicator=0
                        path=0
                        prRed("TIMEOUT!!!tic :%f ,toc:%f"%(tic,toc))
                        continue
                    et.append(t)
                    fp=fp+1
                    # print("Planning Success!!!path:"+str(path))
                    print("Planning Success!!!path:"+str(path))
                    
                else:
                    sp=0
                    indicator=0
                    while indicator==0 and sp<10 and path !=0:
                        toc = time.clock()
                        if (toc-tic) > TIMEOUT :
                            indicator=0
                            path=0
                            prRed("TIMEOUT!!!tic :%f ,toc:%f"%(tic,toc))
                            break
                        sp=sp+1	# 默认重新规划10次
                        # print("Planning Faild!!!RePlanning %d.th......." % sp)
                        print("Planning Faild!!!RePlanning %d.th......." % sp)
                        path=replan_path(path,torch.from_numpy(goal),obs) #replanning at coarse level
                        if path !=0:
                            path=lvc(path)
                            indicator=feasibility_check(path)
                
                            if indicator==1:
                                toc = time.clock()
                                t=toc-tic
                                if t > TIMEOUT:
                                    prRed("TIMEOUT!!!tic :%f ,toc:%f"%(tic,toc))
                                    indicator=0
                                    path=0
                                    break
                                et.append(t)
                                fp=fp+1
                                if len(path)<20:
                                    # print ("new_path:"+str(path))
                                    prGreen("new_path:"+str(path))
                                else:
                                    # print("may be path found,  try again!!!")	
                                    print("may be path found,  try again!!!")
                    if indicator == 0:
                        # print("path not found, dont worry")	
                        prRed("path not found, dont worry")
                        path = 0

          
    # tot.append(et)

    f = open("test1.txt","a")
    print ("total paths")
    print (tp)
    print ("feasible paths")
    print (fp)
    print("time:")
    # print(et)
    print(sum(et)/len(et))
    # print(tot)
    prGreen("total paths: " + str(tp))
    prGreen("feasible paths: " + str(fp))
    prGreen("feasible paths percent: " + str(fp/tp))
    f.write(str(fp/tp)+"\n")
    
    prGreen("avg time: " + str(sum(et)/len(et)))
    f.write("avg time: " + str(sum(et)/len(et)))
    f.write("\n")
    f.close()
# env = generateEnv(0,10).reshape(2000,3)
# env = pc_normalize(env)
# decoder_env = getDecodeEnvPointCloud(torch.from_numpy(env.ravel()),[1,0,0])
# drawPointCloud([env,])
with torch.no_grad():
    env_path = "data/replace3/"
    envs = np.load(env_path+"obs.npy").astype(np.float32)
    obs_center = np.load(env_path+"obs_center.npy").astype(np.float32)
    obs_collision = np.load(env_path+"obs_collision.npy").astype(np.float32)
    init_state = np.load(env_path+"initstate.npy").astype(np.float32)

    for ep in range(10):
       
        # mlp_path = "models/source/3d/mlp/cae_encoder%d_mlp_100_4000_PReLU_ae_dd_final.pkl" % (ep+1)
        mlp_path = "models/source/3d/mlp/cae_offsettimes03_encoder%d_mlp_100_4000_PReLU_ae_dd_final.pkl" % (ep+1)
        # encoder_path = "models/source/3d/encoder/cae_encoder%d.pkl" % (ep+1)
        encoder_path = "models/source/3d/encoder/cae_offset03_encoder%d.pkl" % (ep+1)
        print("use:%s,%s"%(mlp_path,encoder_path))
        f = open("test1.txt","a")
        f.write("use:%s\n"%(env_path))
        f.write("use:%s,%s\n"%(mlp_path,encoder_path))
        f.close()
        mlp.load_state_dict(torch.load(mlp_path))
        encoder.load_state_dict(torch.load(encoder_path))
        generatePath()
