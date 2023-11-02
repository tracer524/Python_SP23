import time, tkinter, math, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani



#各种函数
#简单线性屏障
def linebarrier():
    global barrier, barrierN, barrierS, barrierE, barrierW, barrierNE, barrierNW, barrierSE, barrierSW
    clearBarriers()
    for x in range(int((height/2)-8), int((height/2)+8)):
        barrier[x, int(height/2)] = True
    BarrierDirection()
    BarrierImage()
    BarrierImage()
#圆形障碍物
def roundbarrier():
    global barrier
    clearBarriers()
    xc = int(height/2)   #圆心x坐标
    yc = int(height/2)   #圆心y坐标
    for x in range(-10, 10):
        for y in range(-round(math.sqrt(100-x**2)-1), round(math.sqrt(100-x**2)-1)):
            barrier[y+yc, x+xc] = True
    BarrierDirection()
    BarrierImage()
#矩形障碍物
def rectanglebarrier():
    global barrier
    clearBarriers()
    xc = int(height/2)   #几何中心x坐标
    yc = int(height/2)   #几何中心y坐标
    for x in range(-10, 10):
        for y in range(-5, 5):
            barrier[y+yc, x+xc] = True
    BarrierDirection()
    BarrierImage()
#楔形障碍物
def wedgebarrier():
    global barrier
    clearBarriers()
    xc = int(height/2)   #几何中心x坐标
    yc = int(height/2)   #几何中心y坐标
    for x in range(-10, 20):
        for y in range(-5, round(5-x/2)):
            barrier[y+yc, x+xc] = True
    BarrierDirection()
    BarrierImage()
#平板边界
def flatbarrier():
    global barrier
    clearBarriers()
    for y in range(50, width-50):
        barrier[int(height/2)-16, y] = True
        barrier[int(height/2)+16, y] = True
    BarrierDirection()
    BarrierImage()
#二维翼型
def Two_dimensional_airfoil():
    global barrier
    clearBarriers()
    xc = int(height/2)   #几何中心x坐标
    yc = int(height/2)   #几何中心y坐标
    xi = {}
    yi = {}
    i = 0
    with open('data/Two_dimensional_airfoil.txt', 'r') as file: #调用已经设置好的障碍点坐标
        for line in file:
            words = line.split(' ')
            xi[i] = int(words[0]) - 10
            yi[i] = -int(words[1]) + 9
            i = i + 1
    for i in range(len(xi)):
            barrier[yi[i]+yc, xi[i]+xc] = True
    BarrierDirection()
    BarrierImage()
#清除原有障碍物
def clearBarriers():
    global barrier
    for y in range(0, height):
        for x in range(0, width):
                barrier[y, x] = False
#障碍物各方向的点（用于计算碰撞后的速度）
def BarrierDirection():
    global barrierN, barrierS, barrierE, barrierW, barrierNE, barrierNW, barrierSE, barrierSW
    barrierN = np.roll(barrier,  1, axis=0)		#障碍物以北的点
    barrierS = np.roll(barrier, -1, axis=0)		#障碍物以南的点
    barrierE = np.roll(barrier,  1, axis=1)		#障碍物以东的点
    barrierW = np.roll(barrier, -1, axis=1)		#障碍物以西的点
    barrierNE = np.roll(barrierN,  1, axis=1)    #障碍物东北的点
    barrierNW = np.roll(barrierN, -1, axis=1)    #障碍物西北的点
    barrierSE = np.roll(barrierS,  1, axis=1)    #障碍物东南的点
    barrierSW = np.roll(barrierS, -1, axis=1)    #障碍物西南的点

#初始速度提高
def speedup():
    global u0
    if u0 < 0.09:
        u0 = u0 + 0.01
    else:
        u0 = 0.1
        print("为防止溢出，速度已达最大值。")
    print("当前初始流速为：%f" % u0)
#初始速度减小
def speeddown():
    global u0
    if u0 > -0.09:
        u0 = u0 - 0.01
    else:
        u0 = -0.1
        print("为防止溢出，速度已达最大值。")
    print("当前初始流速为：%f" % u0)

#黏度增加
def viscosityup():
    global viscosity
    viscosity = viscosity + 0.005
    print("当前流体黏度为：%f" % viscosity)
#黏度减小
def viscositydown():
    global viscosity
    if viscosity > 0:
        viscosity = viscosity - 0.005
    else:
        print("显然，流体的黏度最低为0，即理想流体。")
    print("当前流体黏度为：%f" % viscosity)
  

#以下采用D2Q9模型计算流体运动（二维空间，九个离散速度）
#初始条件：流体向右稳定流动（若不定义，则下次运行时将从上次运行的结果继续运行）
def Initial_Condition():
    global rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW
    n0 = w0 * (np.ones((height,width)) - 1.5*u0**2)	
    nN = w1 * (np.ones((height,width)) - 1.5*u0**2)
    nS = w1 * (np.ones((height,width)) - 1.5*u0**2)
    nE = w1 * (np.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    nW = w1 * (np.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    nNE = w2 * (np.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    nSE = w2 * (np.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    nNW = w2 * (np.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    nSW = w2 * (np.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW		#宏观速度
    ux = (nE + nNE + nSE - nW - nNW - nSW) / rho				#沿x方向的宏观速度
    uy = (nN + nNE + nNW - nS - nSE - nSW) / rho				#沿y方向的宏观速度

#经碰撞后，每个格内的粒子的新速度:
def collision():
	global rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW
	rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
	ux = (nE + nNE + nSE - nW - nNW - nSW) / rho
	uy = (nN + nNE + nNW - nS - nSE - nSW) / rho
	ux2 = ux * ux
	uy2 = uy * uy
	u2 = ux2 + uy2
	omu215 = 1 - 1.5*u2
	uxuy = ux * uy
    #以上几项由于需要重复使用，故提前算好
	n0 = (1-omega)*n0 + omega * w0 * rho * omu215
	nN = (1-omega)*nN + omega * w1 * rho * (omu215 + 3*uy + 4.5*uy2)
	nS = (1-omega)*nS + omega * w1 * rho * (omu215 - 3*uy + 4.5*uy2)
	nE = (1-omega)*nE + omega * w1 * rho * (omu215 + 3*ux + 4.5*ux2)
	nW = (1-omega)*nW + omega * w1 * rho * (omu215 - 3*ux + 4.5*ux2)
	nNE = (1-omega)*nNE + omega * w2 * rho * (omu215 + 3*(ux+uy) + 4.5*(u2+2*uxuy))
	nNW = (1-omega)*nNW + omega * w2 * rho * (omu215 + 3*(-ux+uy) + 4.5*(u2-2*uxuy))
	nSE = (1-omega)*nSE + omega * w2 * rho * (omu215 + 3*(ux-uy) + 4.5*(u2-2*uxuy))
	nSW = (1-omega)*nSW + omega * w2 * rho * (omu215 + 3*(-ux-uy) + 4.5*(u2+2*uxuy))
	#在左端的流体强制稳定向右流动（无需设置0、N和S分量）:
	nE[:,0] = w1 * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nW[:,0] = w1 * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNE[:,0] = w2 * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSE[:,0] = w2 * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nNW[:,0] = w2 * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	nSW[:,0] = w2 * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    
#将所有粒子沿其运动方向移动一步的函数：
def streaming():
	global nN, nS, nE, nW, nNE, nNW, nSE, nSW
	nN  = np.roll(nN,   1, axis=0)	# axis = 0 表示沿垂直方向; +表示方向为上
	nNE = np.roll(nNE,  1, axis=0)
	nNW = np.roll(nNW,  1, axis=0)
	nS  = np.roll(nS,  -1, axis=0)
	nSE = np.roll(nSE, -1, axis=0)
	nSW = np.roll(nSW, -1, axis=0)
	nE  = np.roll(nE,   1, axis=1)	# axis = 1 表示沿水平方向；+表示方向为右
	nNE = np.roll(nNE,  1, axis=1)
	nSE = np.roll(nSE,  1, axis=1)
	nW  = np.roll(nW,  -1, axis=1)
	nNW = np.roll(nNW, -1, axis=1)
	nSW = np.roll(nSW, -1, axis=1)

#边界处理（反弹格式）
def BoundProcess():
	nN[barrierN] = nS[barrier]
	nS[barrierS] = nN[barrier]
	nE[barrierE] = nW[barrier]
	nW[barrierW] = nE[barrier]
	nNE[barrierNE] = nSW[barrier]
	nNW[barrierNW] = nSE[barrier]
	nSE[barrierSE] = nNW[barrier]
	nSW[barrierSW] = nNE[barrier]


#计算旋度:
def curl(ux, uy):
	return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

#在绘制动画时每一帧调用的函数：
def Fluid_Animation(arg):					#arg是帧号，本程序不需要，但定义此函数时必须写上
    global startTime, pause
    if not pause:
        for step in range(100):				#动画的步长
            collision()
            streaming()
            BoundProcess()
    Fluid_Image.set_array(curl(ux, uy))
    return (Fluid_Image, barrierImage)		#返回要重新绘制的图形元素

#显示障碍物
def BarrierImage():
    global barrierImage
    bImageArray = np.zeros((height, width, 4), np.uint8)	#生成RGBA图像
    bImageArray[barrier,3] = 255							#在屏障位置设置alpha=255
    barrierImage = plt.imshow(bImageArray, origin='lower', interpolation='none')

#暂停动画
def Pause_Animation():
    global pause
    pause ^= True



#正片开始
#定义常数:
height = 80							#画布高度
width = 300                         #画布宽度
viscosity = 0.005					#黏度
omega = 1 / (3*viscosity + 0.5)		#松弛系数
u0 = 0.05							#初始流入速度
pause = False                       #用于决定动画是否暂停
#lattice-Boltzmann权系数，由于后面反复使用，先提前算好数值：
w0 = 4.0/9.0
w1   = 1.0/9.0
w2  = 1.0/36.0

#将当前工作路径更改成py文件的路径，以便使用相对路径调用数据
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#用布尔数组初始化屏障，有障碍物的地方设为True，没有则设为False:
barrier = np.zeros((height,width), dtype = bool)
BarrierDirection()

#初始化流场
Initial_Condition()

#图形和动画：
Fluid_Figure = plt.figure(figsize=(8,3))
Fluid_Image = plt.imshow(curl(ux, uy), origin='lower', norm=plt.Normalize(-.1,.1), 
									cmap='RdBu_r', interpolation='none')
BarrierImage()

#绘制动画
startTime = time.perf_counter
animate = ani.FuncAnimation(Fluid_Figure, Fluid_Animation, interval=1, blit=True)
#animate.save("animation.gif", writer='pillow')#将生成动画导出为gif文件
#animate.save('animation_Animation.mp4', writer='ffmpeg', fps=30)#将生成动画导出为mp4文件
plt.show()

# 生成窗口类型对象
win = tkinter.Tk()
# 窗口属性的设置
win.title("简单二维流场模拟")
scrnW = win.winfo_screenwidth()
scrnH = win.winfo_screenheight()
win.geometry('500x620+%d+%d' % (scrnW/2-925, scrnH/2-350))

#控件设置
label1 = tkinter.Label(win, text="基本设置", bg='black', fg='white')
label1.place(x=50, y=20)
reset = tkinter.Button(win, text="重置流场",  width = 20, height = 2, command = Initial_Condition)
reset.place(x=50, y=50)
pauseanimation = tkinter.Button(win, text="暂停/继续",  width = 20, height = 2, command = Pause_Animation)
pauseanimation.place(x=250, y=50)

label2 = tkinter.Label(win, text="参数设置", bg='black', fg='white')
label2.place(x=50, y=140)
speed1 = tkinter.Button(win, text="初始流速增大",  width = 20, height = 2, command = speedup)
speed1.place(x=50, y=170)
speed2 = tkinter.Button(win, text="初始流速减小",  width = 20, height = 2, command = speeddown)
speed2.place(x=250, y=170)
viscosity1 = tkinter.Button(win, text="流体黏度增大",  width = 20, height = 2, command = viscosityup)
viscosity1.place(x=50, y=250)
viscosity2 = tkinter.Button(win, text="流体黏度减小",  width = 20, height = 2, command = viscositydown)
viscosity2.place(x=250, y=250)

label3 = tkinter.Label(win, text="屏障设置", bg='black', fg='white')
label3.place(x=50, y=340)
shape1 = tkinter.Button(win, text="线性屏障",  width = 20, height = 2, command = linebarrier)
shape1.place(x=50, y=370)
shape2 = tkinter.Button(win, text="圆形屏障",  width = 20, height = 2, command = roundbarrier)
shape2.place(x=250, y=370)
shape3 = tkinter.Button(win, text="矩形屏障",  width = 20, height = 2, command = rectanglebarrier)
shape3.place(x=50, y=450)
shape4 = tkinter.Button(win, text="楔形屏障",  width = 20, height = 2, command = wedgebarrier)
shape4.place(x=250, y=450)
shape5 = tkinter.Button(win, text="平板边界",  width = 20, height = 2, command = flatbarrier)
shape5.place(x=50, y=530)
shape6 = tkinter.Button(win, text="二维翼型",  width = 20, height = 2, command = Two_dimensional_airfoil)
shape6.place(x=250, y=530)

# 窗口显示
win.mainloop()