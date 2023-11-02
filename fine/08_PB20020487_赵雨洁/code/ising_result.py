# %%
import random
import os
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# %%


class Pipe(object):
    def __init__(self,node1,node2,distribution,seed):
        self.node1=node1
        self.node2=node2#表示两端的节点
        if distribution=='uniform':
            random.seed(seed)#固定种子有利于程序复现、算法优化和序参数研究
            self.value=random.random()
        
        #0代表断开， 1代表连通
        self.state=0

    def isboundary(self):
        return self.node1.group!=self.node2.group

    def __repr__(self):
        return f'{{"node1":{self.node1.coord},\n"node2":{self.node2.coord},\n"value":{self.value},\n"state":{self.state}}}'

class Node(object):
    def __init__(self,n,size,dimension,nodes,pipes):
        
        self.coord=n
        self.nodes=nodes
        self.pipes=pipes
        self.group=-1
        self.vp=[-1]*dimension
    def __repr__(self):
        return f'{{"coord":{self.coord},\n"color":{self.color},\n"group":{self.group},\n"p":{self.p},\n"}}'
    

class Group(object):
    def __init__(self,node_set,index=-1):
        self.set=node_set
        self.index=index

    def __len__(self):
        return len(self.set)
    
    def merge(self,group2):#进行群与群之间的融合
        if len(self)<len(group2):
            self,group2=group2,self

        for node in group2.set:
           
            node.group=self.index
        
        self.set=self.set | group2.set


        return group2.index

     

class Network(object):
    def __init__(self,size,dimension,seed,distribution='uniform'):
   
        self.group_index=0
        self.p=0
        self.save_num=1
    
        self.nodes={}
        self.pipes={}
        self.groups={}

        

        print('Generating Nodes...')
        begin=[0]*dimension
        last=[size]*dimension
        #通过ndim_grid函数使得函数进行填充
        if(dimension!=1):
            nodes=ndim_grid(begin,last)
        else: nodes=np.linspace(0,size-1,size)
        index=0
        #用一维数组来表示n维格点，其中每一格点可以用类似进制的方法得到具体的坐标
        for n in nodes:
            node=Node(n,size,dimension,self.nodes,self.pipes)
            self.nodes[index]=node
            index+=1

        print('Nodes Generated.')

        print('Dying...')
        print('Dyed.')
        
        print('Generating Groups and Pipes...')
        
        for index in range(len(self.nodes)):
            group=self.register_group(Group({self.nodes[index]}))
            self.groups[group].index=group
            self.nodes[index].group=group
            #利用整除和求余数得到坐标
            for dim in range(0,dimension):
                t=int(((index) %(size**(dim+1)))/(size**dim))
                if t!=0:
                    pipe=Pipe(self.nodes[index],self.nodes[index-size**dim],distribution,seed)
                    self.pipes[pipe.value]=pipe
                    node.vp[dim]=pipe.value
                    seed+=pipe.value
        self.values=list(self.pipes.keys())
        self.values.sort()

        print('Init Complete.')
    

    def register_group(self,group):
        self.groups[self.group_index]=group
        self.group_index+=1
        return self.group_index-1

    def groups(self):
        groups=set()
        for node in self.nodes.values():
            groups.add(node.group)
        return groups

    def max_group_size(self):
        size=0
        for group in self.groups:
            gsize=len(self.groups[group])
            if gsize>size:
                size=gsize
        return size

    def connect(self,pipe):
        pipe.state=1
        if pipe.isboundary():
            pop_index=self.groups[pipe.node1.group].merge(self.groups[pipe.node2.group])
            self.groups.pop(pop_index)
            return 1
        return 0

            
    def p_increase(self,newp):
        boundary_pipe_num=0
        self.p=newp
        worklist,self.values=divide(newp,self.values)
        for value in worklist:
            boundary_pipe_num+=self.connect(self.pipes[value])
        return boundary_pipe_num 

def divide(p,value_list):
    i=0
    for value in value_list:
        if p<=value:
            break
        i+=1
    return value_list[:i], value_list[i:]
def ndim_grid(start,stop):

    ndims = len(start)
    L = [np.arange(start[i],stop[i]) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

# %%
net=Network(2,2,4)

# %%
def analyze(size,dim,num,step): 
    ps=np.linspace(0,1,step)

    max_size=[0]*step

    for i in range(num):#取十个种子求平均值提高准确性
        net=Network(size,dim,i)
        t=0
        for p in ps:
            net.p_increase(p)
            max_size[t]+=net.max_group_size()/(num*size**dim)#归一化
            t+=1
    
    
    return ps,max_size

# %%
def draw_group(size):#画group数量和最大群size
    plt.figure(figsize=(7, 7))

    plt.xlim(0, 1,1)
    plt.ylim(0, 1,1)

    plt.grid()
    p_record=[]
   
    ps,max_size1=analyze(1000,1,10,101)
    ps,max_size2=analyze(100,2,10,101)
    ps,max_size3=analyze(10,3,10,101)
    plt.plot(ps,max_size1,label='1D')
    plt.plot(ps,max_size2,label='2D')
    plt.plot(ps,max_size3,label='3D')
    plt.legend()
    plt.savefig(r'Ising_result.jpeg',dpi=300)

    


# %%
draw_group(50)

# %%



