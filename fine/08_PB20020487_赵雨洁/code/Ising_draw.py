# %%
import random
import os
import time
import numpy as np
from PIL import Image

# %%
class IsingNode(object):
    def __init__(self,i,j,index,seed):
        self.i=i
        self.j=j
        self.coord=(i,j)
        random.seed(seed)
        self.color=random.choice((-1,1))
        self.index=index
        self.status=0
        self.neighbors=((i-1,j),(i+1,j),(i,j-1),(i,j+1))
        self.group=None

    def __repr__(self):
        return f"{{'coord':{(self.i,self.j)},'index':{self.index},'status':{self.status}}}"

# %%
class IsingGroup(object):
    def __init__(self,node):
        self.colors={
            1:0,
            -1:0
        }
        self.color=node.color
        self.colors[node.color]+=1
        self.nodes={node}
        node.status=node.color
        node.group=self

    def __iter__(self):
        return self.nodes.__iter__()

    def get_color(self):
        if self.colors[1]>=self.colors[-1]:
            return 1
        else:
            return -1

    def add(self,node,num=1):
        node.group=self
        self.colors[node.color]+=1
        self.nodes.add(node)
        if num ==1 and self.color!=self.get_color():
            self.color=self.get_color()
            for node in self:
                node.status=self.color

    def merge(self,groups):
        for group in groups:
            for node in group:
                self.add(node,-1)
        self.color=self.get_color()
        for node in self:
            node.status=self.color
        
    def __repr__(self):
        return self.nodes.__repr__()

# %%
class IsingNetwork(object):
    def __init__(self,m,n,seed,path,interval):
        self.path=path
        if not os.path.exists(path):
            os.mkdir(path)
        self.m=m
        self.n=n
        self.volume=m*n
        self.nodes={}
        self.net={}
        self.state=0
        self.interval=interval
        index_list=list(range(m*n))
        random.seed(seed)
        random.shuffle(index_list)
        for i in range(m):
            for j in range(n):
                index=index_list[i*m+j]
                self.net[(i,j)]=IsingNode(i,j,index,seed)
                self.nodes[index]=self.net[(i,j)]
                seed+=0.01
        self.color_dict={
            1:[0,0,255],
            -1:[255,0,0],
            0:[0,0,0]
        }

    def get_node(self,coord):
        i=coord[0]
        j=coord[1]
        return self.net[(i%self.m,j%self.n)]

    def get_neighbors_groups(self,coord):
        groups=[]
        for coord in self.get_node(coord).neighbors:
            node=self.get_node(coord)
            if node.group and not node.group in groups:
                groups.append(node.group)
        return groups

    def open_node(self,node):
        groups=self.get_neighbors_groups(node.coord)
        if len(groups)==0:
            group=IsingGroup(node)
        elif len(groups)==1:
            groups[0].add(node)
        else:
            groups[0].add(node)
            groups[0].merge(groups[1:])

    def save_img(self):
        path=self.path
        img=[[0]*self.n for i in range(self.m)]
        for node in self.nodes.values():
            img[node.i][node.j]=self.color_dict[node.status]
        img=Image.fromarray(np.asarray(np.array(img),np.uint8))
        img.save(path+str(self.state)+'.png')

    def __next__(self):
        if self.state<self.volume:
            self.open_node(self.nodes[self.state])
            percentage=self.state/(self.volume)
            self.state+=1
            if self.state%self.interval==0:
                self.save_img()
                return 1
        else:
            raise StopIteration()

    def __iter__(self):
        return self
        

# %%
def png_to_gif(png_path,gif_name,duration):
    """png合成gif图像"""
    frames = []
    # 返回文件夹内的所有静态图的列表
    png_files = os.listdir(png_path)
    png_list=[]
    for f in png_files:
        if f[-3:]=='png':
            png_list.append(int(f[:-4]))
    png_list.sort()

    # 读取文件内的静态图
    for png in png_list:
        frame = Image.open(os.path.join(png_path,str(png)+'.png'))
        frames.append(frame)
    frames[0].save(gif_name,format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

# %%
path='img_ising/'

# %%
net=IsingNetwork(100,100,1,path,100)

for i in net:
    if i==10:
        time.sleep(0.1)

# %%
png_to_gif(path,'ising.gif',100)

# %%



