# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from test1 import Network
import os
# %%
#matplotlib画图，但性能较低
color_list1=[
        'r',
        'g',
        'b',
        'c',
        'm',
        'y',
        'k'
        

    ]
size=20
seed=4
net=Network(size,size,seed,color_list=color_list1)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
xticks=np.arange(-1,size,1)
yticks=np.arange(-1,size,1)
ax.set_xlim(-1, size,1)
ax.set_ylim(-1, size,1)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.grid()
time_template = 'p = %.1f'
time_text = ax.text(0.9, 0.9,'', transform=ax.transAxes)

group_template = 'group number = %d'
group_text = ax.text(0.9, 0.86,'', transform=ax.transAxes)



def update(p):

    net.p_increase(p)
    for node in net.nodes.values():
        ax.scatter(node.i,node.j,s=20, c=net.color_dict[node.color])
    time_text.set_text(time_template % (p))
    group_text.set_text(group_template % (len(net.groups)))

ani=animation.FuncAnimation(fig, update, np.linspace(0,1,11),interval=100)
ani.save('matplotlib2D.gif', writer='imagemagick', fps=2)


def analyze(size,num,step): 
    ps=np.linspace(0,1,step)
    group_num=[0]*step
    group_numdel=[0]*step
    max_size=[0]*step

    for i in range(num):#取十个种子求平均值提高准确性
        net=Network(size,size,i,color_list=color_list1)
        t=0
        for p in ps:
            net.p_increase(p)
            group_num[t]+=len(net.groups)/num
            max_size[t]+=net.max_group_size()/num
            t+=1
    for i in range(len(group_num)-1):#求group数量的变化率
        group_numdel[i]=group_num[i]-group_num[i+1]
    
    return ps,group_num,max_size,group_numdel

# %%
def draw_group(ps,group_num,max_size,size):#画group数量和最大群size
    plt.figure(figsize=(7, 7))

    plt.xlim(0, 1,1)
    plt.ylim(-1, size*size,1)

    plt.grid()

    plt.plot(ps,group_num,c='r',label='group number')
    plt.plot(ps,max_size,c='b',label='max group size')
    plt.legend()
    plt.savefig(r'group.jpeg',dpi=300)

def draw_changerate(ps,group_numdel,size):
    
    plt.figure(figsize=(7, 7))

    plt.xlim(0, 1,1)
    plt.ylim(-1, 200,1)

    plt.grid()

    plt.plot(ps,group_numdel,c='g',label='group number change rate')
    plt.legend()
    plt.savefig(r'changerate.jpeg',dpi=300)


ps,gn,ms,nd=analyze(100,10,101)
draw_group(ps,gn,ms,100)
draw_changerate(ps,nd,100)

def draw_boundry(size): 
    net=Network(size,size,4)
    boundary_pipe_num_list=[]
    for i in np.linspace(0,1,101):
        boundary_pipe_num_list.append(net.p_increase(i))
    plt.plot(np.linspace(0,1,100),boundary_pipe_num_list[1:],label='boundry number')
    plt.legend()
    plt.savefig(r'boundry.jpeg',dpi=300)
draw_boundry(200)

def dtGEN(path,num):#读取gif图的最后修改时间来判断获取每一帧图像的时间
    tlist=[]
    for i in range(1,num+2):
        tlist.append(os.path.getmtime(path+str(i)+'.png'))
    
    ttlist=[]
    dtlist=[]
    lt=0
    for t in tlist:
        if lt!=0:
            dtlist.append(t-lt)
            ttlist.append(t-tlist[0])
        lt=t

    return dtlist,ttlist

def compare(num):#比较两种算法的优劣和程序的运行时间
    l1,t1=dtGEN('imgs/',num)
    l2,t2=dtGEN('imgs3/',num)
    x=np.linspace(0,1,num,endpoint=False)
    plt.figure(figsize=(10,5))
    ax1=plt.subplot(1,2,1)
    ax2=plt.subplot(1,2,2)

    ax1.plot(x,l1,label='method1')
    ax1.plot(x,l2,label='method2')

    ax2.plot(x,t1,label='method1')
    ax2.plot(x,t2,label='method2')

    plt.savefig(r'compare.jpeg',dpi=300)


