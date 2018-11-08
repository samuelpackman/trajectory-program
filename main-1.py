import numpy as np
import math
import matplotlib.pyplot as plt

def Call_Vars():
    global K,No_Go,delta
    K = ['air','force','gravity','angle',"delta"]
    No_Go=input('What do you want this graph to be a function of? : ')
    for i in range(5):
        if K[i]!= No_Go:
            K[i] =   float(input(str(K[i]) + ': '))
    if type(K[3])==float:
        K[3]*=np.pi/180
def vPrime(velocity):
    return np.array([0,-1 * K[2]]) - K[0] * np.linalg.norm(velocity) * velocity
def move(position,velocity):
    return position + K[4] * velocity
def r(p):
    if p=="air":
        return [1,0,True,"resistance"]
    elif p=="force":
        return [100,1,True,"(N)"]
    elif p=="gravity":
        return [20,2,False,"(m/s/s)"]
    elif p=="angle":
        return [np.pi/2,3,True,"(rad)"]
def Option_1():
    arguments =[i*(r(No_Go)[0])/200 for i in range(1,201)]
    if r(No_Go)[2]:
        arguments=[0]+arguments
    values=[]
    for i in arguments:
        K[r(No_Go)[1]]=i
        v = K[1]*np.array([np.cos(K[3]), np.sin(K[3])])
        x = np.array([0,0])
        while x[1] >= 0:
            x = (move(x,v) + move(x,move(v,vPrime(v))))/2
            v = move(v,vPrime(v))
        values.append(x[0])
    plt.title(No_Go.capitalize() + ' vs Distance')
    plt.ylim(top= 4/3 * max(values))
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right = max(arguments))
    plt.xlabel(No_Go.capitalize() + ' ' + r(No_Go)[3])
    plt.ylabel('Distance (m)')
    line = plt.plot(arguments,values,lw=2)
    plt.show()
def Option_2():

    v = K[1]*np.array([np.cos(K[3]), np.sin(K[3])])
    total_x = np.array([0,0])
    x = np.array([0,0])

    while x[1] >= 0:
        x = (move(x,v) + move(x,move(v,vPrime(v))))/2
        v = move(v,vPrime(v))
        total_x=np.vstack((total_x,x))
    total_x=np.rot90(total_x)
    times = K[4]*np.arange(len(total_x[0]))

    if No_Go=="trajectory":
        label="Trajectory of Launch"
        xlabel="x-coordinates"
        ylabel="y-coordinates"
        xlist=total_x[1]
        ylist=total_x[0]
    elif No_Go=="x v time":
        label="x of Time"
        xlabel="time"
        ylabel="x-coordinates"
        xlist=times
        ylist=total_x[1]
    elif No_Go=="y v time":
        label="y of Time"
        xlabel="time"
        ylabel="y-coordinate"
        xlist=times
        ylist=total_x[0]
    plt.title(label)
    plt.ylim(top = 4/3 * max(ylist))
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right = max(xlist))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    line = plt.plot(xlist,ylist, lw=2)
    plt.show()

Call_Vars()
if No_Go in K:
    Option_1()
else:
    Option_2()


"""def Plot_1():
    import matplotlib.pyplot as plt
    plt.title(No_Go.capitalize() + ' vs Distance')
    plt.ylim(top= 4/3 * max(values))
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right = max(arguments))
    plt.xlabel(No_Go.capitalize() + ' ' + r(No_Go)[3])
    plt.ylabel('Distance (m)')
    line = plt.plot(arguments,values,lw=2)
    plt.show()"""
"""def Plot_2():
    import matplotlib.pyplot as plt
    D = total_x[1][-1]
    plt.title('Graph of Launch')
    plt.ylim(top = 3/4 * D)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right = D)
    line, = plt.plot(total_x[1],total_x[0], lw=2)
    plt.show()"""
