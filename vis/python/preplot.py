import sys
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
# import dynamo as dyn
from scipy.integrate import odeint,ode
from scipy import optimize
import astropy.constants as cons

# positive direction: VR(outwards), Vphi(counterclockwise)
def pol2car(r,theta): # transform polar coordinate to cartesian
    lenx = len(r)
    leny = len(theta)
    x1 = zeros((lenx,leny))
    x2 = zeros((lenx,leny))
    for i in range(lenx):
        for j in range(leny):
            x1[i,j] = r[i]*cos(theta[j])
            x2[i,j] = r[i]*sin(theta[j])
    return x1,x2

def car2pol(x1,x2):
    lenx = len(x1)
    leny = len(x2)
    r = zeros((leny,lenx))
    theta = zeros((leny,lenx))
    
    for j in range(leny):
        for i in range(lenx):
            r[j,i] = sqrt(x1[i]**2+x2[j]**2)
            theta[j,i] = angle(x1[i] + 1j*x2[j])
            if(theta[j,i]<0):
                theta[j,i]+=2*pi
#             print(x1[i],x2[j])
    return r,theta

def v_Intplp2c_xy(x1_origin,x2_origin,x1_expect,x2_expect,vx,vy):
    #needed gird in Cartesian, transformed to polar first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    r = zeros((leny,lenx))
    theta = zeros((leny,lenx))
    array_expect = zeros((lenx*leny,2))
    
    for j in range(leny):
        for i in range(lenx):
            # transfromed to polar coordinate
            r[j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2)
            theta[j,i] = angle(x1_expect[i] + 1j*x2_expect[j])
            if(theta[j,i]<0):
                theta[j,i]+=2*pi
            # transfromed to 1-D array to interpolate 
            array_expect[lenx*j+i,0] = r[j,i]
            array_expect[lenx*j+i,1] = theta[j,i]
            
    #do interpolation in Polar coordinate
    fvx = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vx,fill_value=None,bounds_error = False)
    fvy = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vy,fill_value=None,bounds_error = False)
    vx_polar = fvx(array_expect).reshape((leny,lenx))
    vy_polar = fvy(array_expect).reshape((leny,lenx))
    
    #cast to cartesian
    vx_cartesian = vx_polar*cos(theta)-vy_polar*sin(theta)
    vy_cartesian = vx_polar*sin(theta)+vy_polar*cos(theta)
    
    return vx_cartesian,vy_cartesian

def v_Intplp2c_xz(x1_origin,x2_origin,x1_expect,x2_expect,vx,vy):
    #needed gird in Cartesian, transformed to polar first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    r = zeros((leny,lenx))
    theta = zeros((leny,lenx))
    array_expect = zeros((lenx*leny,2))
    
    for j in range(leny):
        for i in range(lenx):
            # transfromed to polar coordinate
            r[j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2)
            theta[j,i] = pi/2 - angle(x1_expect[i] + 1j*x2_expect[j])
            # transfromed to 1-D array to interpolate 
            array_expect[lenx*j+i,0] = r[j,i]
            array_expect[lenx*j+i,1] = theta[j,i]
            
    #do interpolation in Polar coordinate
    fvx = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vx,fill_value=None,bounds_error = False)
    fvy = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vy,fill_value=None,bounds_error = False)
    vx_polar = fvx(array_expect).reshape((leny,lenx))
    vy_polar = fvy(array_expect).reshape((leny,lenx))
    
    #cast to cartesian
    vx_cartesian = vx_polar*sin(theta)+vy_polar*cos(theta)
    vy_cartesian = vx_polar*cos(theta)-vy_polar*sin(theta)
    
    return vx_cartesian,vy_cartesian

def v_Intpl_Sph2car(x1_origin,x2_origin,x3_origin,x1_expect,x2_expect,x3_expect,v1,v2,v3):
    #needed gird in Cartesian, transformed to spherical first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    lenz = len(x3_expect)
    r = zeros((lenz,leny,lenx))
    theta = zeros((lenz,leny,lenx))
    phi = zeros((lenz,leny,lenx))
    array_expect = zeros((lenx*leny*lenz,3))
    
    for k in range(lenz):
        for j in range(leny):
            for i in range(lenx):
                # transfromed to polar coordinate
                r[k,j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2+x3_expect[k]**2)
                theta[k,j,i] = arccos(x3_expect[k]/r[k,j,i])
                phi[k,j,i] = angle(x1_expect[i] + 1j*x2_expect[j])
                if(phi[k,j,i]<0):
                    phi[k,j,i]+=2*pi
                # transfromed to 1-D array to interpolate 
                array_expect[(leny*lenx)*k + lenx*j+i,0] = r[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,1] = theta[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,2] = phi[k,j,i]
            
    #do interpolation in Polar coordinate
    fvx = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),v1,fill_value=0,bounds_error = False)
    fvy = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),v2,fill_value=0,bounds_error = False)
    fvz = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),v3,fill_value=0,bounds_error = False)
    vx_sph = fvx(array_expect).reshape((lenz,leny,lenx))
    vy_sph = fvy(array_expect).reshape((lenz,leny,lenx))
    vz_sph = fvz(array_expect).reshape((lenz,leny,lenx))
    
    #cast to cartesian
    vx_cartesian = sin(theta)*cos(phi)*vx_sph + cos(theta)*cos(phi)*vy_sph - sin(phi)*vz_sph
    vy_cartesian = sin(theta)*sin(phi)*vx_sph + cos(theta)*sin(phi)*vy_sph + cos(phi)*vz_sph
    vz_cartesian = cos(theta)*vx_sph - sin(theta)*vy_sph
    
    return vx_cartesian,vy_cartesian,vz_cartesian

def scaler_Intpl_Sph2car(x1_origin,x2_origin,x3_origin,x1_expect,x2_expect,x3_expect,rho):
    #needed gird in Cartesian, transformed to spherical first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    lenz = len(x3_expect)
    r = zeros((lenz,leny,lenx))
    theta = zeros((lenz,leny,lenx))
    phi = zeros((lenz,leny,lenx))
    array_expect = zeros((lenx*leny*lenz,3))
    
    for k in range(lenz):
        for j in range(leny):
            for i in range(lenx):
                # transfromed to polar coordinate
                r[k,j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2+x3_expect[k]**2)
                theta[k,j,i] = arccos(x3_expect[k]/r[k,j,i])
                phi[k,j,i] = angle(x1_expect[i] + 1j*x2_expect[j])
                if(phi[k,j,i]<0):
                    phi[k,j,i]+=2*pi
                # transfromed to 1-D array to interpolate 
                array_expect[(leny*lenx)*k + lenx*j+i,0] = r[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,1] = theta[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,2] = phi[k,j,i]
            
    #do interpolation in Polar coordinate
    f0 = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),rho,fill_value=0.0,bounds_error = False)
    rho_car = f0(array_expect).reshape((lenz,leny,lenx))
    
    return rho_car


# Ref:https://en.wikipedia.org/wiki/Finite_difference_coefficient

def dfdx_2pts(x,y):
    num = len(x)
    dx = abs(x[1]-x[0])
    dfdx = zeros(num)
    for i in range(num):
        if(i>=1):
            dfdx[i] = (y[i]-y[i-1])/dx
        else:
            dfdx[i] = (y[i+1]-y[i])/dx
    return dfdx

def dfdx_5pts(x,y):
    num = len(x)
    dx = abs(x[1]-x[0])
    dfdx = zeros(num)
    for i in range(num):
        if((i>=2) and ((num-i)>=3)):
            dfdx[i] = (-y[i+2]+8*y[i+1]-8*y[i-1]+y[i-2])/(12*dx)
        elif(i<2):
            dfdx[i] = (-25/12*y[i]+4*y[i+1]-3*y[i+2]+4/3*y[i+3]-1/4*y[i+4])/dx
        else:
            dfdx[i] = -(-25/12*y[i]+4*y[i-1]-3*y[i-2]+4/3*y[i-3]-1/4*y[i-4])/dx
    return dfdx

def dfdx_7pts(x,y):
    num = len(x)
    dx = abs(x[1]-x[0])
    dfdx = zeros(num)
    for i in range(num):
        if((i>=3) and ((num-i)>=4)):
            dfdx[i] = (-1/60*y[i-3]+3/20*y[i-2]-3/4*y[i-1]+3/4*y[i+1]-3/20*y[i+2]+1/60*y[i+3])/dx
        elif(i<3):
            dfdx[i] = (-49/20*y[i]+6*y[i+1]-15/2*y[i+2]+20/3*y[i+3]-15/4*y[i+4]+6/5*y[i+5]-1/6*y[i+6])/dx
        elif(num-i<4):
            dfdx[i] = -(-49/20*y[i]+6*y[i-1]-15/2*y[i-2]+20/3*y[i-3]-15/4*y[i-4]+6/5*y[i-5]-1/6*y[i-6])/dx
    return dfdx


def curl_in_polar_rlog(r,theta,vr,vtheta):
    numr= len(r)
    numtheta = len(theta)
    logr = log10(r)
    rvtheta = zeros((numr,numtheta))
    dfdlogr = zeros((numr,numtheta))
    dthetadvr = zeros((numr,numtheta))
    
    for i in range(numr):
        rvtheta[i,:] = r[i]*vtheta[i,:]
        dthetadvr[i,:] = dfdx_7pts(theta,vr[i,:]) /(r[i])
        
    for j in range(numtheta):
        dfdlogr[:,j] = dfdx_7pts(logr,rvtheta[:,j])
    for i in range(numr):
        dfdlogr[i,:] /=r[i]**2
    return 1/log(10)*dfdlogr -dthetadvr


def search_separatrix2(diff,saddle,xs,r_dev,theta):
    Rd = 0
    index = 0
    
    dt = 0.1
    t_total =40
    f_intgl = ode(diff).set_integrator('dopri5',first_step = 1e-4)
    while(Rd<xs and index < 100):
        xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
        f_intgl.set_initial_value(array(saddle)+array(xy_dev))
        f_intgl.t = 0
        while (f_intgl.successful() and (f_intgl.t < t_total) and abs(f_intgl.y[0])<xs and abs(f_intgl.y[1])<ys):
            f_intgl.integrate(dt+f_intgl.t)
        pos = f_intgl.y
        Rd = sqrt(pos[0]**2+pos[1]**2)
        index +=1
        
    pos0 = pos
    d = 0
    if(index == 1):
        while(Rd>xs and d<2*ys  and index <100):
            xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
            f_intgl.set_initial_value(array(saddle)+array(xy_dev))
            f_intgl.t = 0
            while (f_intgl.successful() and (f_intgl.t < t_total) and abs(f_intgl.y[0])<xs and abs(f_intgl.y[1])<ys):
                f_intgl.integrate(dt+f_intgl.t)
            pos = f_intgl.y
            Rd = sqrt(pos[0]**2+pos[1]**2)
            index +=1
            d = sqrt(sum((pos-pos0)**2))
            print(index,pos)
            pos0 = pos
            
#             print(d)
    return index-1


def search_separatrix(diff,saddle,r_dev,theta):
    d_max = 2e-2
    dt = 0.1
    d = 0
    dir_change = False
    index = 0
    f_intgl = ode(diff).set_integrator('dopri5',first_step = 1e-4)
    # first loop to define dx0
    xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
    f_intgl.set_initial_value(array(saddle)+array(xy_dev))
    f_intgl.t = 0
    
    while (f_intgl.successful() and (d < d_max)):
        f_intgl.integrate(dt+f_intgl.t)
        d = sqrt(sum((f_intgl.y-saddle)**2))
    dx0 = f_intgl.y-saddle
    index +=1
    print(dx0)
    
    # the following loops
    while(dir_change == False):
        d = 0
        xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
        f_intgl.set_initial_value(array(saddle)+array(xy_dev))
        f_intgl.t = 0
        while (f_intgl.successful() and (d < d_max)):
            f_intgl.integrate(dt+f_intgl.t)
            d = sqrt(sum((f_intgl.y-saddle)**2))
        dx = f_intgl.y-saddle
        print(index,dx)
        if(dx[0]*dx0[0]<0 or dx[1]*dx0[1]<0):
            dir_change = True
        index += 1
    return index-1


def face_f_2_cos(x2min,x2max,cell_width_ratio,num_face):
    x = linspace(0,1,num_face)
    w = arccos(1-x)/(pi/2)
    tmp = w*(x2max-x2min) + x2min
    
    return tmp

def face_f_2_power(x2min,x2max,cell_width_ratio,num_face):
    x = linspace(0,1,num_face)
    w = (x)**(1/3)
    tmp = w*(x2max-x2min) + x2min
    
    return tmp

def face_f_1_log(x1min,x1max,cell_width_ratio,num_face):
    x = linspace(0,1,num_face)
    w = log10(x1min) + x*(log10(x1max)-log10(x1min))
    tmp = 10**w
    return tmp