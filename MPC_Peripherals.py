import socket
import sys
from marvelmindmod import MarvelmindHedge
import numpy as np

def velPID(velerror, velerrorprev, accelprev, Isum, tdelt, steer):
    Pgain = 38000.0
    Igain = 540.0
    Dgain = 2000.0
    filtgain = 0.75
    steerband = steer**2
    if steerband <= 0.25:
        steerband = 0.0
    steergain = steerband * 1300.0
    P = Pgain * velerror
    I = (Igain*velerror) + Isum
    accel = ((velerror-velerrorprev)/tdelt)
    accelfilt = (filtgain*accelprev)+((1-filtgain)*accel)
    D = Dgain * accelfilt
    throttle = P + I + D + steergain
    throttle = np.clip(throttle,0,60000)
    Isum = I
    return throttle, Isum, accelfilt, P, I, D

def marvelmindSetup():
    hedge = MarvelmindHedge(tty = 'COM4', adr=None, debug=False) # create MarvelmindHedge thread, modify tty to match port
    if (len(sys.argv)>1):
        hedge.tty= sys.argv[1]
    hedge.start() # start thread
    return hedge

def marvelmindGetPos(hedge):
    X = 0
    Y = 0
    V = 0

    hedge.dataEvent.wait(1)
    hedge.dataEvent.clear()
    if (hedge.positionUpdated):
            # hedge.print_position()
            X = hedge.position()[1]
            Y = hedge.position()[2]
            V = hedge.position()[3]

    return X, Y, V

def socketSetup():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind(('192.168.8.126', 9999))
    s.listen()
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established")
    return clientsocket

def trackDefine():
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

    return rottrack, delt

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist = np.sum((nodes-node)**2, axis=1)
    n = np.argmin(dist)
    return n

def trim(value):
    if value>=1:
        value=1
    elif value<=-1:
        value=-1
    return value