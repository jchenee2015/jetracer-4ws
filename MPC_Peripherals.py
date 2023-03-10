import socket
import sys
from marvelmindmod import MarvelmindHedge
import numpy as np
from scipy.signal import lsim

def velPID(velerror, velerrorprev, accelprev, Isum, tdelt, steer):
    Pgain = 38000.0
    Igain = 540.0
    Dgain = 2000.0
    filtgain = 0.75
    steerband = steer**2
    if steerband <= 0.25:
        steerband = 0.0
    steergain = steerband * 1300.0
    # Pgain = 27000.0
    # Igain = 275.0
    # Dgain = 1500.0
    # filtgain = 0.75
    # steerband = steer**2
    # if steerband <= 0.25:
    #     steerband = 0
    # steergain = steerband * 1300.0
    P = Pgain * velerror
    I = (Igain*velerror) + Isum
    # D = Dgain * ((delt-deltprev)/(tdelt))
    accel = ((velerror-velerrorprev)/tdelt)
    accelfilt = (filtgain*accelprev)+((1-filtgain)*accel)
    D = Dgain * accelfilt
    # D = Dgain * (delt-deltprev)
    # throttle = throttle + steergain + P + I + D # look into signs again, read articles
    throttle = P + I + D + steergain
    throttle = np.clip(throttle,0,60000)
    # throttle = throttle + steergain
    # throttle = -0.172834485
    # deltprev = delt
    Isum = I
    return throttle, Isum, accelfilt, P, I, D

# def velPIDlaplace(delt, Isum, tdelt, steer):
#     # Pgain = 0.0000000005
#     # Igain = 0.0
#     # Dgain = 0.0
#     Pgain = 25000.0
#     Igain = 275.0
#     Dgain = 3500.0
#     tau = 0.1
#     num = [(Dgain/tau),0]
#     den = [1,(1/tau)]
#     steerband = steer**2
#     if steerband <= 0.25:
#         steerband = 0
#     steergain = steerband * 4500.0
#     # throttle = 0
#     # delt = 1.0-vx
#     # if np.sign(delt) != np.sign(deltprev):
#     #     Isum = 0
#     P = Pgain * delt[-1]
#     I = (Igain*delt[-1]) + Isum
#     # D = Dgain * ((delt-deltprev)/(tdelt))
#     # D = Dgain * ((delt[1]-delt[0])/(tdelt))
#     tout, D, xout = lsim((num,den),U=delt,T=tdelt)
#     throttle = P + I + D[-1] + steergain
#     throttle = np.clip(throttle,0,15000)
#     # throttle = throttle + steergain
#     # throttle = -0.172834485
#     # deltprev = delt
#     Isum = I
#     return throttle, Isum, P, I, D[-1]

def marvelmindSetup():
    hedge = MarvelmindHedge(tty = 'COM4', adr=None, debug=False) # create MarvelmindHedge thread
    if (len(sys.argv)>1):
        hedge.tty= sys.argv[1]
    hedge.start() # start thread
    return hedge

def marvelmindGetPos(hedge):
    X = 0
    Y = 0
    V = 0
    # Ximu = 0
    # Yimu = 0
    # Zimu = 0
    # Vximu = 0
    # Vyimu = 0
    # Vzimu = 0
    # Aximu = 0
    # Ayimu = 0
    # Azimu = 0
    hedge.dataEvent.wait(1)
    hedge.dataEvent.clear()
    if (hedge.positionUpdated):
            # hedge.print_position()
            X = hedge.position()[1]
            Y = hedge.position()[2]
            V = hedge.position()[3]
    # if (hedge.fusionImuUpdated):
    #         # hedge.print_imu_fusion()
    #         # Ximu = hedge.imu_fusion()[0]
    #         # Yimu = hedge.imu_fusion()[1]
    #         # Zimu = hedge.imu_fusion()[2]
    #         Vximu = hedge.imu_fusion()[7]
    #         Vyimu = hedge.imu_fusion()[8]
    #         # Vzimu = hedge.imu_fusion()[9]
    #         # Aximu = hedge.imu_fusion()[10]
    #         # Ayimu = hedge.imu_fusion()[11]
    #         # Azimu = hedge.imu_fusion()[12]

    return X, Y, V

def socketSetup():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind(('192.168.8.126', 9999))
    s.listen()
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established")
    return clientsocket

def trackDefinelarge():
    xshift = 3.5
    yshift = 2.2

    scale = 1.6

    theta = np.radians(90.0)
    r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    quant = 400
    delt = (np.pi*scale)/(quant-1)

    top = np.empty((quant,2))

    for i in range(len(top)):
        ang = (np.pi/(quant-1))*i
        top[i,0] = np.cos(ang) * scale
        top[i,1] = np.sin(ang) * scale
    bottom = -top
    gap = 0
    gapcount = 0
    while gap < (scale*1.0):
        gapcount += 1
        gap += delt
    top[:,1] += gap
    bottom[:,1] -= gap
    left = np.empty(((2*gapcount)-1,2))
    left[:,0] = -scale
    for i in range(len(left)):
        left[i,1] = (gap-delt)-(i*delt)
    right = -left
    track = np.vstack((top,left,bottom,right))

    rottrack = track
    for i in range(len(track)):
        rottrack[i,:] = r.dot(track[i,:])

    rottrack[:,0] = rottrack[:,0] + xshift
    rottrack[:,1] = (rottrack[:,1] + yshift)

    # topang = np.ones(len(top[:,0]))
    # topang[0] = 0.5
    # topang[len(topang)-1] = 0.5
    # leftang = np.zeros(len(left[:,0]))
    # rottrackang = np.hstack((topang,leftang,topang,leftang))

    return rottrack, delt #, rottrackang, scale

def trackDefine():
    xshift = 3.5
    yshift = 2.2

    scale = 0.86

    theta = np.radians(90.0)
    r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    quant = 200
    delt = (np.pi*scale)/(quant-1)

    top = np.empty((quant,2))

    for i in range(len(top)):
        ang = (np.pi/(quant-1))*i
        top[i,0] = np.cos(ang) * scale
        top[i,1] = np.sin(ang) * scale
    bottom = -top
    gap = 0
    gapcount = 0
    while gap < (scale*0.75):
        gapcount += 1
        gap += delt
    top[:,1] += gap
    bottom[:,1] -= gap
    left = np.empty(((2*gapcount)-1,2))
    left[:,0] = -scale
    for i in range(len(left)):
        left[i,1] = (gap-delt)-(i*delt)
    right = -left
    track = np.vstack((top,left,bottom,right))

    rottrack = track
    for i in range(len(track)):
        rottrack[i,:] = r.dot(track[i,:])

    rottrack[:,0] = rottrack[:,0] + xshift
    rottrack[:,1] = (rottrack[:,1] + yshift)

    # topang = np.ones(len(top[:,0]))
    # topang[0] = 0.5
    # topang[len(topang)-1] = 0.5
    # leftang = np.zeros(len(left[:,0]))
    # rottrackang = np.hstack((topang,leftang,topang,leftang))

    return rottrack, delt #, rottrackang, scale

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist = np.sum((nodes-node)**2, axis=1)
    n = np.argmin(dist)
    # control = np.sqrt((((node[0])-(nodes[n,0]))**2)+(((node[1])-(nodes[n,1]))**2))
    # if np.sqrt((node[0]**2)+(node[1]**2)) - np.sqrt((nodes[n,0]**2)+(nodes[n,1]**2)) >= 0:
    #     sign = 1
    # else:
    #     sign = -1
    # control = control * sign
    return n

def trim(value):
    if value>=1:
        value=1
    elif value<=-1:
        value=-1
    return value