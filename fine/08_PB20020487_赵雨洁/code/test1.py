import random
import os
import time
import numpy as np
from PIL import Image
'''
面向对象的编程
为了实现目的，一共定义四个类：Pipe，Node，Group，Network
Pipe类:表示管道，属性node1，node2表示指向管道两端的节点；属性value在0～1之间，根据给定概率分布随机产生的值；
Node类：表示节点
Group类：表示节点所在的群，每个群内的节点互相联通，群与群之间不联通。
Network类：表示网络，负责控制群与群之间的行为
'''
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
    def __init__(self,i,j,nodes,pipes):
        self.i=i
        self.j=j
        self.coord=(i,j)#节点坐标，不同的表现形式使用不同的方法（更便捷）
        self.nodes=nodes
        self.pipes=pipes
        self.color=-1
        self.group=-1
        self.rp=-1
        self.dp=-1 #初始化

    def __repr__(self):
        return f'{{"coord":{self.coord},\n"color":{self.color},\n"group":{self.group},\n"rp":{self.rp},\n"dp":{self.dp}}}'

class Group(object):
    def __init__(self,node_set,color,index=-1):
        self.set=node_set
        self.color=color
        self.index=index

    def __len__(self):
        return len(self.set)
    
    def merge(self,group2):#进行群与群之间的融合
        if len(self)<len(group2):
            self,group2=group2,self

        for node in group2.set:
            node.color=self.color
            node.group=self.index
        
        self.set=self.set | group2.set


        return group2.index

        
    def __repr__(self):
        return f'{{"color":{self.color},"set":{self.set}}}'

class Network(object):
    def __init__(self,m,n,seed,distribution='uniform',color_num=7,color_list=[]):
        self.scale=(m,n)

        self.group_index=0
        self.p=0
        self.save_num=1
    
        self.nodes={}
        self.pipes={}
        self.groups={}

        if len(color_list):
            color_num=len(color_list)
            self.color_dict=dict(zip(range(color_num),color_list))
        else:
            self.color_dict=dict(zip(range(color_num),range(color_num)))

        print('Generating Nodes...')
        for i in range(m):
            for j in range(n):
                node=Node(i,j,self.nodes,self.pipes)
                self.nodes[(i,j)]=node

        print('Nodes Generated.')

        print('Dying...')
        self.dye(color_num,seed)
        print('Dyed.')
        
        print('Generating Groups and Pipes...')
        for node in self.nodes.values():
            group=self.register_group(Group({node},node.color))
            self.groups[group].index=group
            node.group=group
            if node.i != m-1:
                pipe=Pipe(node,self.get_node(node.i+1,node.j),distribution,seed)
                self.pipes[pipe.value]=pipe
                node.dp=pipe.value
                seed+=pipe.value
            if node.j != n-1:
                pipe=Pipe(node,self.get_node(node.i,node.j+1),distribution,seed)
                self.pipes[pipe.value]=pipe
                node.rp=pipe.value
                seed+=pipe.value
        self.values=list(self.pipes.keys())
        self.values.sort()

        print('Init Complete.')
    
    def dye(self,color_num,seed):
        for node in self.nodes.values():
            clist=list(range(color_num))
            (i,j)=node.coord
            if i!=0:
                clist.remove(self.get_node(i-1,j).color)
            if j!=0:
                if self.get_node(i,j-1).color in clist:
                    clist.remove(self.get_node(i,j-1).color)
            random.seed(seed)
            node.color=random.choice(clist)
            seed+=node.color/2


    def register_group(self,group):
        self.groups[self.group_index]=group
        self.group_index+=1
        return self.group_index-1

    def get_node(self,i,j):
        return self.nodes[(i,j)]

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

  

    def save_img(self,path):
        if not os.path.exists(path):
            os.mkdir(path)
        img=[[0]*self.scale[1] for i in range(self.scale[0])]
        for node in self.nodes.values():
            img[node.i][node.j]=self.color_dict[node.color]
        img=Image.fromarray(np.asarray(np.array(img),np.uint8))
        img.save(path+str(self.save_num)+'.png')
        self.save_num+=1


    
    def __repr__(self):
        repr=''
        m=self.scale[0]
        n=self.scale[1]
        for i in range(m):
            for j in range(n):
                repr+=' '+str(self.color_dict[self.get_node(i,j).color])+' '
                if j !=n-1:
                    if self.pipes[self.get_node(i,j).rp].state:
                        repr+='-'
                    else:
                        repr+=' '
            repr+='\n'
            if i !=m-1:
                for j in range(self.scale[1]):
                    if self.pipes[self.get_node(i,j).dp].state:
                        repr+=' | '
                    else:
                        repr+='   '
                    if j!=n-1:
                        repr+=' '
                repr+='\n'
        return repr

def divide(p,value_list):
    i=0
    for value in value_list:
        if p<=value:
            break
        i+=1
    return value_list[:i], value_list[i:]

if __name__ =='__main__':
    color_list=[
        '\033[31m◉\033[0m',
        

        '\033[34m◉\033[0m',
       
    ]

    net=Network(15,15,4,color_list=color_list)
    for p in np.linspace(0,1,101):
        os.system('clear')
        print('p={:.2f}'.format(p))
        net.p_increase(p)
        print(net)
        time.sleep(0.2)

    net=Network(15,15,4,color_list=color_list)
    for p in np.linspace(0,1,101):
        os.system('clear')
        print('p={:.2f}'.format(p))
        net.p_increase(p)
        print(net)
        time.sleep(0.2)


