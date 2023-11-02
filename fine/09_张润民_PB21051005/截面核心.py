import numpy as np
import turtle

def to_standard_s(s):
    t=[]
    for i in range(0,len(s)):
        if len(s[i])==2:
            t+=[(s[i])]
        elif len(s[i])==3:
            if i==len(s)-1:
                p1=(s[i][0],s[i][1])
                p2=(s[0][0],s[0][1])
                r=s[i][2]
                t+=arc_to_s(p1,p2,r) 
            else:
                p1=(s[i][0],s[i][1])
                p2=(s[i+1][0],s[i+1][1])
                r=s[i][2]
                t+=arc_to_s(p1,p2,r)       
    return t

def get_circle(p0, p1, r):
    if p1[0] == p0[0]:
        y0 = y1 = (p0[1]+p1[1]) / 2
        deltay = (y0-p0[1]) ** 2
        deltax = np.sqrt(r ** 2 - deltay)
        x0 = p1[0] - deltax
        x1 = p1[0] + deltax
    else:
        C1 = (p1[0]**2 + p1[1]**2 - p0[0]**2 - p0[1]**2) / 2 / (p1[0] - p0[0])
        C2 = (p1[1] - p0[1]) / (p1[0] - p0[0])
        A = 1 + C2**2
        B = 2 * (p0[0] - C1) * C2 - 2*p0[1]
        C = (p0[0]-C1)**2 + p0[1]**2 - r**2
        y0 = (-B + np.sqrt(B*B - 4 * A * C)) / 2 / A
        y1 = (-B - np.sqrt(B*B - 4 * A * C)) / 2 / A
        x0 = C1 - C2 * y0
        x1 = C1 - C2 * y1
    if r>0:
        if (p1[0] == p0[0] and p1[1]>p0[1])or p1[0] > p0[0] :
            return [(x0, y0), (x1, y1)]
        else:
            return [(x1, y1),(x0, y0)]
    elif r<0:
        if (p1[0] == p0[0] and p1[1]>p0[1])or p1[0] > p0[0] :
            return [(x1, y1),(x0, y0)]
        else:
            return [(x0, y0), (x1, y1)]       
         
def arc_to_s(p1,p2,r): #把圆弧变成点列（r>0表示逆时针画，r<0表示顺时针画）
    d=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    if r>0:
        a0=2*np.arcsin(d/(2*r))
        c=get_circle(p1, p2, r)[0]
        
        if p1[0]>c[0]:
            a=np.arctan((p1[1]-c[1])/(p1[0]-c[0]))
        elif p1[0]==c[0]:
            if p1[1]>c[1]:
                a=np.pi/2
            else:
                a=-np.pi/2
        else:   
            if p1[1]>c[1]:
                a=np.pi+np.arctan((p1[1]-c[1])/(p1[0]-c[0]))
            else:
                a=-np.pi+np.arctan((p1[1]-c[1])/(p1[0]-c[0]))   
        s=[]
        b=a0/10
        s+=[(p1)]
        for i in range(1,10):
            s+=[((c[0]+r*np.cos(i*b+a)),(c[1]+r*np.sin(i*b+a)))]
        return s 
    else:
        r=-r
        a0=2*np.arcsin(d/(2*r))
        c=get_circle(p1, p2, r)[1]
        
        if p1[0]>c[0]:
            a=np.arctan((p1[1]-c[1])/(p1[0]-c[0]))
        elif p1[0]==c[0]:
            if p1[1]>c[1]:
                a=np.pi/2
            else:
                a=-np.pi/2
        else:   
            if p1[1]>c[1]:
                a=np.pi+np.arctan((p1[1]-c[1])/(p1[0]-c[0]))
            else:
                a=-np.pi+np.arctan((p1[1]-c[1])/(p1[0]-c[0]))         

        s=[]
        b=-a0/10
        s+=[(p1)]
        for i in range(1,10):
            s+=[((c[0]+r*np.cos(i*b+a)),(c[1]+r*np.sin(i*b+a)))]
        return s 
        
def cal_area(vertices): #Gauss's area formula 高斯面积计算 计算截面面积
    A = 0.0
    point_p = vertices[-1]
    for point in vertices:
        A += (point[1]*point_p[0] - point[0]*point_p[1])
        point_p = point
    return abs(A)/2

def cal_centroid(points): #计算质心位置
    A = cal_area(points)
    c_x, c_y = 0.0, 0.0
    point_p = points[-1] # point_p 表示前一节点
    for point in points:
        c_x +=((point[0] + point_p[0]) * (point[1]*point_p[0] - point_p[1]*point[0]))
        c_y +=((point[1] + point_p[1]) * (point[1]*point_p[0] - point_p[1]*point[0]))
        point_p = point

    return c_x / (6*A), c_y / (6*A)

    
def is_tu(p1,p2,s): #判断是否是凸多边形
    t=[]
    if (p1[0]!=p2[0])and(p1[1]!=p2[1]):
        for p in s:
            a=(p[1]-p2[1])/(p1[1]-p2[1])-(p[0]-p2[0])/(p1[0]-p2[0])
            t+=[a]
        while 0 in t:
            t.remove(0)
        for i in range(0,len(t)-1):
            if t[i]*t[i+1]<0:
                return False
        return True
    elif p1[0]==p2[0]:
        for p in s:
            a=p[0]-p1[0]
            t+=[a]
        while 0 in t:
            t.remove(0)
        for i in range(0,len(t)-1):
            if t[i]*t[i+1]<0:
                return False
        return True
    elif p1[1]==p2[1]:
        for p in s:
            a=p[1]-p1[1]
            t+=[a]
        while 0 in t:
            t.remove(0)
        for i in range(0,len(t)-1):
            if t[i]*t[i+1]<0:
                return False
        return True
            
def find_tangent(s): #找切线点集
    t=[]
    for i in range(0,len(s)):
        for j in range(0,len(s)):
            if i!=j:              
                if is_tu(s[i],s[j],s):
                    t+=[(s[i])]
                    break
    return t
        
def  cms_to0(s): #质心移到原点  
    c=cal_centroid(s)

    t=[]
    for i in s:
        t+=[(i[0]-c[0],i[1]-c[1])]
    return t

       
def Iz(s):
    I=0
    for i in range(0,len(s)):
        if i==len(s)-1:
            I+=(s[i][0]*s[0][1]-s[0][0]*s[i][1])*(s[i][0]*s[i][0]+s[i][0]*s[0][0]+s[0][0]*s[0][0])
        else:
            I+=(s[i][0]*s[i+1][1]-s[i+1][0]*s[i][1])*(s[i][0]*s[i][0]+s[i][0]*s[i+1][0]+s[i+1][0]*s[i+1][0])
    return I/12
                              
def Iy(s):
    I=0
    for i in range(0,len(s)):
        if i==len(s)-1:
            I+=(s[i][0]*s[0][1]-s[0][0]*s[i][1])*(s[i][1]*s[i][1]+s[i][1]*s[0][1]+s[0][1]*s[0][1])
        else:
            I+=(s[i][0]*s[i+1][1]-s[i+1][0]*s[i][1])*(s[i][1]*s[i][1]+s[i][1]*s[i+1][1]+s[i+1][1]*s[i+1][1])
    return I/12    

def Iyz(s):
    I=0
    for i in range(0,len(s)):
        if i==len(s)-1:
            I+=(s[i][0]*s[i][0]*s[0][1]*(2*s[i][1]+s[0][1])-s[0][0]*s[0][0]*s[i][1]*(2*s[0][1]+s[i][1])+2*s[i][0]*s[0][0]*(s[0][1]*s[0][1]-s[i][1]*s[i][1]))
        else:
            I+=(s[i][0]*s[i][0]*s[i+1][1]*(2*s[i][1]+s[i+1][1])-s[i+1][0]*s[i+1][0]*s[i][1]*(2*s[i+1][1]+s[i][1])+2*s[i][0]*s[i+1][0]*(s[i+1][1]*s[i+1][1]-s[i][1]*s[i][1]))
    return I/24

def cal_a0(s): #计算转到主惯性的转角（顺时针）
    if Iy(s)!=Iz(s):
        return 0.5*np.arctan(-2*Iyz(s)/(Iy(s)-Iz(s)))
    else:
        if Iyz(s)<0:
            return 0.5*np.pi
        elif Iyz(s)>0:
            return -0.5*np.pi
        else:
            return 0

def to_standard(s):
    t=cms_to0(s)
    u=[]
    a=cal_a0(t)
    for i in t:
        u+=[(i[0]*np.cos(a)+i[1]*np.sin(a),-i[0]*np.sin(a)+i[1]*np.cos(a))]
    return u    

def find_core(s):
    a=cal_a0(cms_to0(s))
    t=to_standard(s)

    u=find_tangent(t)

    v=[]
    w=[]
    iz2=Iz(t)/cal_area(t)
    iy2=Iy(t)/cal_area(t)
    for i in range(0,len(u)):
        if i==len(u)-1:     #计算切线截距
            if u[i][0]!=u[0][0] and u[i][1]!=u[0][1]:
                az=-u[0][0]*(u[i][1]-u[0][1])/(u[i][0]-u[0][0])+u[0][1]
                ay=-u[0][1]*(u[i][0]-u[0][0])/(u[i][1]-u[0][1])+u[0][0]
                v+=[(-iz2/ay,-iy2/az)]
            elif u[i][0]==u[0][0]:
                v+=[(-iz2/u[0][0],0)]
            elif u[i][1]==u[0][1]:
                v+=[(0,-iy2/u[0][1])]                    
        else:
            if u[i][0]!=u[i+1][0] and u[i][1]!=u[i+1][1]:
                az=-u[i+1][0]*(u[i][1]-u[i+1][1])/(u[i][0]-u[i+1][0])+u[i+1][1]
                ay=-u[i+1][1]*(u[i][0]-u[i+1][0])/(u[i][1]-u[i+1][1])+u[i+1][0]
                v+=[(-iz2/ay,-iy2/az)]
            elif u[i][0]==u[i+1][0]:
                v+=[(-iz2/u[i+1][0],0)]
            elif u[i][1]==u[i+1][1]:
                v+=[(0,-iy2/u[i+1][1])]  
    for i in v:
        w+=[(i[0]*np.cos(a)-i[1]*np.sin(a),i[0]*np.sin(a)+i[1]*np.cos(a))]
    return w
         

s1=[(0,0),(400,0),(300,300),(100,300)]
s2=[(120,0),(270,60),(240,180),(90,210),(180,90),(90,120),(30,90)]
s3=[(0,0),(300,0),(300,250,150),(0,250)]
s4=[(-258,276),(-22,276),(-22,-320),(258,-320),(258,-276),(22,-276),(22,320),(-258,320)]
s5=[(250,150),(-250,150),(-250,-150),(250,-150)]
s6=[(50,0),(150,0),(150,200),(200,200),(200,300),(0,300),(0,200),(50,200)]
s7=[(90,120),(210,120),(210,180),(30,180),(30,-180),(210,-180),(210,-120),(90,-120)]
s8=[(100,0,100),(-100,0,100),(-100,-200),(100,-200)]
s9=[(100,100),(-100,100),(-100,-100),(100,-100)]
s10=[(164.16,20.85,19.37),(158.86,59,30.97),(121.48,82.25,-76.92),(90.72,83.46,23.46),(62.37,62.36,24),(35.31,28.9)]


s=s10
t1=cms_to0(to_standard_s(s))
t2=find_core(to_standard_s(s))
print(t2)

turtle.hideturtle()
turtle.penup()
turtle.goto(t2[0])
turtle.pendown()
turtle.fillcolor('green')
turtle.begin_fill()
for i in t2:
    turtle.goto(i)
turtle.goto(t2[0])
turtle.end_fill()

turtle.penup()
turtle.goto(t1[0])
turtle.pendown()
for i in t1:
    turtle.goto(i)
turtle.goto(t1[0])

turtle.penup()
turtle.goto(-400,0)
turtle.pendown()
turtle.goto(400,0)
turtle.penup()
turtle.goto(0,-400)
turtle.pendown()
turtle.goto(0,400)

turtle.mainloop()





