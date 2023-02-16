
import open3d as o3d
import numpy as np


def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0) # 求取点云的中心
    pc = pc - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc / m # 依据长轴将点云归一化到 (-1, 1)
    return pc_normalized,m,centroid
# def show_pointcloud():
#     print("Load a ply point cloud, print it, and render it")
#     # ply_point_cloud = o3d.data.PLYPointCloud()
#     plyname = r't.pts'
#     #读点云
#     pcd = o3d.io.read_point_cloud(plyname,format="xyzrgb")
#     print(pcd)
#     print(np.asarray(pcd.points))
#     #点云显示
#     o3d.visualization.draw_geometries([pcd])
#     #保存点云
#     o3d.io.write_point_cloud("save.pcd", pcd)

def getEnvInput(file):
    obs = np.fromfile(file).astype(np.float32).reshape(-1,3)
    return obs

def getPathInput(file):
    path = np.fromfile(file).astype(np.float32).reshape(-1,3)
    return path
    
def getOBSPointCloudXYZ(file, color):
    obs = getEnvInput(file)
    obs_num = obs.shape[0]
    obs_rgb = np.empty((obs_num,6))
    rgb = np.tile(color,(obs_num,1))
    obs_rgb[:,:3] = obs
    obs_rgb[:,3:] = rgb
    return obs_rgb

def getPathPointCloud(file,color):
    path = getPathInput(file)
    path_num = path.shape[0]
    path_rgb = np.empty((path_num,6))
    rgb = np.tile(color,(path_num,1))
    path_rgb[:,:3] = path
    path_rgb[:,3:] = rgb
    return path_rgb

def getDecodeEnvPointCloud(input,color):
    
    with torch.no_grad():
        input.requires_grad = False
        encoder = Encoder()
        encoder.load_state_dict(torch.load("models/cae_r8_addOffset_encoder.pkl"))
        # encoder.load_state_dict(torch.load("models/cae_r8_encoder.pkl"))
        encoder.eval()
        feature = encoder(input)
        
        decoder = Decoder()
        decoder.load_state_dict(torch.load("models/cae_r8_addOffset_decoder.pkl"))
        # decoder.load_state_dict(torch.load("models/cae_r8_decoder.pkl"))
        decoder.eval()
        output = decoder(feature)
    
    env = output.numpy().reshape(-1,3)
    env_num = env.shape[0]
    rgb = np.tile(color,(env_num,1))
    decode_env = np.empty((env_num,6))
    decode_env[:,:3] = env
    decode_env[:,3:] = rgb
    
    return decode_env

def drawPointCloud(pc):
    pc_list = []
    i = 0
    draw_color = [0,1,0]
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()	#创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
    # render_option.background_color = np.array([0, 0, 0])	#设置背景色（这里为黑色）
    render_option.point_size = 3.0
    for x in pc:
        pc_list.append(o3d.geometry.PointCloud())
        pc_list[i].points = o3d.utility.Vector3dVector(x[:,:3])
        if x.shape[1] == 6:
            draw_color = x[0][3:]
        pc_list[i].paint_uniform_color(draw_color)
        # pc_list[i].poi
        i +=1
    for x in pc_list:
        vis.add_geometry(x)	#添加点云
    
    vis.run()
    
def drawPath(pc):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()	#创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
    # render_option.background_color = np.array([0, 0, 0])	#设置背景色（这里为黑色）
    render_option.point_size = 3.0
    
    pc_list = []
    #绘制场景
    pc_list.append(o3d.geometry.PointCloud())
    pc_list[0].points = o3d.utility.Vector3dVector(pc[0][:,:3])
    pc_list[0].paint_uniform_color([0, 1, 0])
    # 绘制轨迹连线
    pc_list.append(o3d.geometry.LineSet())
    line = []
    for x in range(len(pc[1])-1):
        line.append([x,x+1])
    colors = [[1, 0, 0] for i in range(len(line))]
    pc_list[1].lines = o3d.utility.Vector2iVector(line)
    pc_list[1].colors = o3d.utility.Vector3dVector(colors)
    pc_list[1].points = o3d.utility.Vector3dVector(pc[1][:,:3])
    # 绘制轨迹点
    pc_list.append(o3d.geometry.PointCloud())
    pc_list[2].points = o3d.utility.Vector3dVector(pc[1][:,:3])
    pc_list[2].paint_uniform_color([0,0,1])
    
    # 绘制障碍物边界
    for x in range(len(pc[2])):
        pc_list.append(o3d.geometry.LineSet())
        line = [
            [0,1],
            [0,2],
            [0,4],
            [1,3],
            [1,5],
            [2,3],
            [2,6],
            [3,7],
            [4,5],
            [4,6],
            [5,7],
            [6,7]
        ]
        pc_list[3+x].lines = o3d.utility.Vector2iVector(line)
        colors = [[0, 1, 0] for i in range(len(line))]
        pc_list[3+x].colors = o3d.utility.Vector3dVector(colors)
        pc_list[3+x].points = o3d.utility.Vector3dVector(pc[2][x])
    for x in pc_list:
        vis.add_geometry(x)	#添加点云
    vis.run()

def draw_collision(pc):
    vis = o3d.visualization.Visualizer()
    vis.create_window()	#创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
    # render_option.background_color = np.array([0, 0, 0])	#设置背景色（这里为黑色）
    render_option.point_size = 3.0
    
    pc_list = []
    #绘制场景
    pc_list.append(o3d.geometry.PointCloud())
    pc_list[0].points = o3d.utility.Vector3dVector(pc[0][:,:3])
    pc_list[0].paint_uniform_color([0, 1, 0])
    
    # 绘制障碍物边界
    for x in range(len(pc[1])):
        pc_list.append(o3d.geometry.LineSet())
        line = [
            [0,1],
            [0,2],
            [0,4],
            [1,3],
            [1,5],
            [2,3],
            [2,6],
            [3,7],
            [4,5],
            [4,6],
            [5,7],
            [6,7]
        ]
        pc_list[1+x].lines = o3d.utility.Vector2iVector(line)
        colors = [[0, 1, 0] for i in range(len(line))]
        pc_list[1+x].colors = o3d.utility.Vector3dVector(colors)
        pc_list[1+x].points = o3d.utility.Vector3dVector(pc[1][x])
    for x in pc_list:
        vis.add_geometry(x)	#添加点云
    vis.run()