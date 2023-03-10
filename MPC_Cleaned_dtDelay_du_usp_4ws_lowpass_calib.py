from marvelmindmod import MarvelmindHedge
import time
import sys
import socket
import numpy as np
import mpctools as mpc
from MPC_Peripherals import *


class vehEnv:
    def __init__(self, hedge, T=10,rho=0):
        # System parameters.
        self.p = 9
        self.Nx = 3
        self.Nu = 2
        self.dt = 0.2
        self.dtdelay = 1
        
        self.hedge = hedge
        
        self.track, self.delt = trackDefinelarge()
        
        self.Xnew, self.Ynew, self.vx = marvelmindGetPos(self.hedge)
        # self.vx = 0.0 # np.sqrt(self.Vxnew**2 + self.Vynew**2)
        self.Angnew = 0.0
        self.Xold = self.Xnew
        self.Yold = self.Ynew
        self.vxfilt = self.vx
        self.Angnewfilt = self.Angnew
        self.Angoldfilt = self.Angnew
        self.vxold = 0.0
        self.vellpgain = 0.67
        self.Angold = np.empty(1) * self.Angnew
        self.Timeold = time.time()

        self.xsp = np.zeros((self.p+1,self.Nx))
        self.n = 0
        self.nd = 0
        self.distoff = 0.0
        
        self.T = T
        self.rho = rho
        self.x0 = [self.Xnew, self.Ynew, self.Angnewfilt]

        # Define parameters
        pi = np.pi
        # rho_aero = 1.225
        # mveh = 1500
        # Cd = 0.389
        # radwhl = 0.2159
        # Af= 4
        # Lxf = 1.2
        # Lxr = 1.4
        # mu = 1
        # Iw = 3.8782
        # I = 4192
        # Calpha = -0.08*180/pi
        # sa_max = 10/180*pi
        # g = 9.8
        rho_aero = 1.225
        mveh = 2.2
        Cd = 0.389
        radwhl = 0.035
        Af= 0.019
        Lxf = 0.03
        Lxr = 0.23
        mu = 1.0
        Iw = 0.00003
        I = 0.0265*26.0
        Calpha = -0.08*180/pi
        sa_max = pi*30.0/180.0
        g = 9.8
        self.Tf = 0.048

        self.param_mpc = [rho_aero, mveh, Cd, radwhl, Af, Lxf, Lxr, mu, Iw, I, Calpha, sa_max,  g]

        self.data = np.zeros((1,16))

        self.f = mpc.getCasadiFunc(self.ode, [self.Nx, self.Nu, 1], ["self.x", "self.u", "self.vx"], rk4=True, Delta=self.dt, M=4)

        self.l = mpc.getCasadiFunc(self.stagecost, [self.Nx, self.Nu, self.Nx, self.Nu, 4], ["x", "u", "x_sp", "Du", "gainlist"])

        # self.lb = dict(x=np.array([-np.inf, -np.inf, -np.inf]), u=np.array([-0.30, -0.07]), Du=np.array([-0.10, -0.04]))
        # self.ub = dict(x=np.array([np.inf, np.inf, np.inf]), u=np.array([0.30, 0.07]), Du=np.array([0.10, 0.04]))
        self.lb = dict(x=np.array([-np.inf, -np.inf, -np.inf]), u=np.array([-0.2, -0.04]), Du=np.array([-0.04, -0.01]))
        self.ub = dict(x=np.array([np.inf, np.inf, np.inf]), u=np.array([0.2, 0.04]), Du=np.array([0.04, 0.01]))
        self.udiscrete = np.array([False, False])
        self.N = dict(x=self.Nx, u=self.Nu, t=self.p)

        self.gainrow = rho
        self.gains = np.loadtxt('4DOF_DOEarrayinsidelarge_2_18', delimiter=' , ')
        self.gains = np.vstack(([14.0,16.0,12.0,14.0],self.gains))
        self.gains = np.vstack((self.gains,[14.0,16.0,12.0,14.0]))

    def reset(self,gainnum):
        self.x = self.x0
        self.k = 0
        self.useq = np.zeros([self.p,self.Nu])
        self.xseq = np.zeros([self.p,self.Nx])
        self.t = 0.0
        self.data = np.zeros((1,16))
        self.gainrow = gainnum
                
    def getFeat(self):
        return np.concatenate((self.x,self.xseq[self.k]))

    def veh_rhs(self, x0, t, u0, vx = 0.7):
        param = self.param_mpc
        rho_aero = param[0]
        mveh = param[1]
        Cd = param[2]
        radwhl = param[3]
        Af= param[4]
        Lxf = param[5]
        Lxr = param[6]
        mu = param[7]
        # Iw = param[8]
        I = param[9]
        Calpha = param[10]
        # sa_max = param[11]    
        g = param[12]

        # sigma = 0.0;
        # small_pos_th = 1e-6;

        # initialization
        # vx = x0[1];
        # vy = x0[3];
        # psi = x0[4];
        # r = x0[5];
        # vx = 7.5;
        psi = x0[2]
        # vy = 0.0

        Tf = self.Tf #36.0 #u0[0];
        Tr = 0.0 #u0[1];
        betaf = u0[0] #u0[2];
        betar = u0[1] #u0[3];

        beta = np.arctan((Lxf*np.tan(betar)+Lxr*np.tan(betaf))/(Lxf+Lxr))
        
        # calculating the xdot
        # xdot = np.zeros(6)
        xdot0 = vx*np.cos(psi+beta)
        xdot2 = vx*np.sin(psi+beta)
        xdot3 = ((vx*np.cos(beta)/(Lxr+Lxf))*(np.tan(betaf)-np.tan(betar)))
        xdot=(xdot0,xdot2,xdot3)
        # return xdot
        return xdot

    def ode(self, x, u, vx):
        """ODE right-hand side."""
        # u0 = np.zeros(4)
        # u0[0]=36
        # u0[2]=u
        dxdt = self.veh_rhs(x, 0., u, vx=vx)
        return np.array(dxdt, dtype=object)

    # Stage cost.
    def stagecost(self, x, u, xsp, du, gainlist):
        """Quadratic stage cost."""
        offset = ((x[1]-xsp[1])**2 + (x[0]-xsp[0])**2)
        return 3.0*(offset) + gainlist[0]*(du[0]**2) + gainlist[1]*(du[1]**2) + gainlist[2]*(u[0]**2) + gainlist[3]*(u[1]**2)
        # return 2.0*(x[1] - 4*np.sin(2*pi/50*x[0]))**2 + 3.0*u[0]**2 

    def step(self, action, throttle,P,I,D): 
        a,b,c = marvelmindGetPos(self.hedge)
        self.timediff = time.time()-self.Timeold
        self.Timeold = time.time()
        if (a != 0.0):
            self.Xnew = a
            self.Ynew = b
            # self.Vxnew = c
            # self.Vynew = d
            # self.vx = np.sqrt(self.Vxnew**2 + self.Vynew**2)
            self.vx = c # np.sqrt((self.Xnew-self.Xold)**2+(self.Ynew-self.Yold)**2)/self.timediff
            if self.vxfilt >= 1.35 and np.abs(self.vx - self.vxfilt)>=0.35:
                self.vx = self.vxfilt
            
            self.vxfilt = (self.vellpgain*self.vxold)+((1-self.vellpgain)*self.vx)
            self.vxold = self.vxfilt
            self.Angnew = np.arctan2(self.Ynew-self.Yold,self.Xnew-self.Xold)
            self.Xold = self.Xnew
            self.Yold = self.Ynew
            self.Angold = np.roll(self.Angold,1)
            self.Angold[0] = self.Angnew
            self.Angnewfilt = self.Angnew #np.average(Angold)
            self.Angoldfilt = self.Angnewfilt

            # self.Vlong = np.sqrt((self.vx**2)/(1+((4.0/(np.pi**2))*(self.Yawnew**2))))
            # self.Vlat = np.sqrt(self.vx**2 - self.Vlong**2) * np.sign(self.Yawnew)
            
            self.dist = self.vxfilt * self.dt
            self.points = int(np.rint(self.dist/self.delt))
            self.n = closest_node((self.Xnew,self.Ynew),self.track)
            self.nd = self.n + (self.points * self.dtdelay)
            if self.nd >= len(self.track):
                self.nd = self.nd - len(self.track)
            self.distoff = np.sqrt((self.Xnew-self.track[self.n,0])**2 + (self.Ynew-self.track[self.n,1])**2)
            if ((self.Xnew-3.5)**2 + (self.Ynew+2.2)**2) >= ((self.track[self.n,0]-3.5)**2 + (self.track[self.n,1]+2.2)**2):
                sign = 1
            else:
                sign = -1
            self.distoff = self.distoff*sign
            
            self.xsp = np.zeros((self.p+1,self.Nx))
            # self.xsp[:,0] = self.Xnew
            # self.xsp[:,1] = self.Ynew
            for i in range(self.p+1):
                val = self.nd+(i*self.points)
                while val >= len(self.track):
                    val = val-len(self.track)
                self.xsp[i,0] = self.track[val,0]
                self.xsp[i,1] = self.track[val,1]
            # print(self.xsp)
        
        self.beta0 = np.arctan((self.param_mpc[5]*np.tan(self.useq[0,1])+self.param_mpc[6]*np.tan(self.useq[0,0]))/(self.param_mpc[5]+self.param_mpc[6]))
        self.psi0 = self.Angnewfilt-self.beta0
        self.x0 = np.array([self.Xnew, self.Ynew, self.psi0]) # x position, y position, lat velocity, yaw angle, yaw vel
        # print(self.x0)
        # x0 = np.array([0, 0, 0, 0, 0])
        if self.dtdelay == 0:
            self.x0d = self.x0
        else:
            self.xd = np.zeros((self.dtdelay+1, self.Nx))
            self.xd[0,:] = self.x0
            for t in range(self.dtdelay):
                self.xd[t + 1,:] = np.squeeze(self.f(self.xd[t,:], self.useq[t,:], self.vxfilt))
            self.x0d = self.xd[self.dtdelay,:]

        self.x = np.zeros((self.p + 1, self.Nx))
        self.u = np.zeros((self.p, self.Nu))
        self.x[0,:] = self.x0d

        for t in range(self.p):
            self.x[t + 1,:] = np.squeeze(self.f(self.x[t,:], self.u[t,:], self.vxfilt))

        guess = dict(x=self.x, u=self.u)

        # Create controller.

        self.sp = {"x" : self.xsp}

        if action > 0:
            # cont = mpc.nmpc(f, l, N, self.x0d, lb, ub, guess, sp=dict(x=self.xsp), extrapar=dict(vx=self.vxfilt, ulast=self.useq[0], xsp=self.xsp[0,:]), funcargs={"f" : ["x","u","vx"], "l" : ["x","u","xsp","ulast"]}, udiscrete=udiscrete) # only thing to change is self.x becomes feedback measurements
            cont = mpc.nmpc(self.f, self.l, self.N, self.x0d, self.lb, self.ub, guess, sp=self.sp, uprev=self.useq[0,:], extrapar=dict(vx=self.vxfilt, gainlist=self.gains[self.gainrow,:]), funcargs={"f" : ["x","u","vx"], "l" : ["x","u","x_sp","Du","gainlist"]}, udiscrete=self.udiscrete)
            cont.solve()
            self.useq = cont.vardict["u"]
            self.xseq = cont.vardict["x"]
            self.k = 0
        else:
            self.k+=1  
            if self.k > self.p-1:
                self.k = self.p-1
        print(self.u)
        self.u = self.useq[self.k,:] # 0 here would be time triggered rather than event
        
        # t = np.linspace(self.t, self.t+dt, 10)
        # aux = integrate.odeint(veh_rhs, self.x, t, args=(u,))
        # self.x = aux[-1] # - this will come from marvelmind
        # self.t += self.dt 
        
        # # Reward should be calculated using current state as forward Euler is using. 
        # reward = torch.tensor(stagecost(self.x, u)*self.dt*(-1)-self.rho*action)
        
        # next_state = torch.tensor(self.getFeat())
       
        # current_time = torch.tensor(self.t)
        
        # if self.t >= self.T:
        #     done = True
        # else:
        #     done = False
        print(self.xseq)
        self.data = np.append(self.data,[[self.Xnew,self.Ynew,self.vxfilt,self.vx,throttle,self.x0d[0],self.x0d[1],self.Angnewfilt,self.n,self.distoff,self.u[0],self.u[1],self.timediff,P,I,D]],axis = 0)

        return self.u , self.data, self.distoff, self.vxfilt, self.timediff

def steerloop(Isum,velerrorprev,accelprev,throttle,P,I,D):
    steering , data, distoff, vxfiltered, timediff = env.step(1,throttle,P,I,D)
    # steeringmapf = -trim(steering[0]*3.34)
    # steeringmapr = -trim(steering[1]*3.34)
    steeringmapf = trim((-176.36*(steering[0]**5))-(34.572*(steering[0]**4))+(14.168*(steering[0]**3))+(1.9324*(steering[0]**2))-(2.9107*steering[0]))
    steeringmapr = trim((-176.36*(steering[1]**5))-(34.572*(steering[1]**4))+(14.168*(steering[1]**3))+(1.9324*(steering[1]**2))-(2.9107*steering[1]))
    steeringAngintf = int((steeringmapf+1.0) * 100.0)
    steeringAngintr = int((steeringmapr+1.0) * 100.0)
    steerbytesf = steeringAngintf.to_bytes(2,'little')
    steerbytesr = steeringAngintr.to_bytes(2,'little')
    velerror = 1.6 - vxfiltered
    throttle, Isum, accelprev, P, I, D = velPID(velerror,velerrorprev,accelprev,Isum,timediff,steeringmapf)
    velerrorprev = velerror
    throttleint = int(throttle)
    throttlebytes = throttleint.to_bytes(6,'little')
    steerbytes = b''.join([steerbytesf,steerbytesr,throttlebytes])#steerbytesf + steerbytesr + throttlebytes
    s.send(steerbytes)

    return data, distoff, Isum,velerrorprev,accelprev,throttle,P,I,D

# Initial condition, bounds, etc.

# clientsocket = socketSetup()
# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(("192.168.8.126",9090))
# s.bind(("192.168.8.214", 9999))

hedge = marvelmindSetup()

Isum = 0.0
accelprev = 0.0
velerrorprev = 0.0
throttle = 0.0
P = 0.0
I = 0.0
D = 0.0

datalist = []
env = vehEnv(hedge, T=10,rho=0)

time.sleep(0.1)

for i in range(27):
    env.reset(0)
    endtime = time.time() + 20
    while time.time() <= endtime:
        data, distoff, Isum,velerrorprev,accelprev,throttle,P,I,D = steerloop(Isum,velerrorprev,accelprev,throttle,P,I,D)

    env = vehEnv(hedge, T=10,rho=i)
    env.reset(i)
    endtime = time.time() + 45
    while time.time() <= endtime:
        data, distoff, Isum,velerrorprev,accelprev,throttle,P,I,D = steerloop(Isum,velerrorprev,accelprev,throttle,P,I,D)
        if distoff >= 0.40:
            break
    
    datalist.append(data)

for i in range(27):
    np.savetxt('mpckinemdata_1_2_20_1dt_4ws_caltest_gain{}'.format(i), datalist[i], delimiter = ' , ')