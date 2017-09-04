
from curiosity_proj import *
from lib_robotis_hack import *
import numpy as np 

moves = np.linspace(-1.6,1.6,4)
#Actions = [(x,y,z,w) for x in moves for y in moves for z in moves for w in moves]
#Actions = move left, move right, stay the same 
#Actions = [(x,y,z,w) for x in range(-1,2) for y in range(-1,2) for z in range (-1,2) for w in range (-1,2)]#move from -1.6 to 1.6
Actions = [(0,0,0,0), (0.5,0.5,0.5,0.5), (-0.5,-0.5,-0.5,-0.5)]
States = np.linspace(-1.6, 1.6, num=10, endpoint=True, retstep=False, dtype=None)


D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AL01QFD9",baudrate=1000000)#	tty.usbserial-AL01QFD9	
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
s2 = Robotis_Servo(D,s_list[1])
s3 = Robotis_Servo(D,s_list[2])
s4 = Robotis_Servo(D,s_list[3])  

s1.move_angle(0)
s2.move_angle(0)
s3.move_angle(0)
s4.move_angle(0)

def returnServos():
        return s1,s2,s3,s4

def init(s1,s2,s3,s4): #state initialization might be in env
        return(s1.read_angle(), s2.read_angle(),s3.read_angle(),s4.read_angle())
    
def takeAction(action):
    #s1,s2,s3,s4 = state
        ang1,ang2,ang3,ang4 = init(s1,s2,s3,s4)
        anglist = [ang1,ang2,ang3,ang4]
        #print (s1.read_angle(),s2.read_angle(),s3.read_angle(),s4.read_angle())  
        servos = [s1,s2,s3,s4]
        #print(ang1,ang2,ang3,ang4)
        for i in range (0,4):
                newangle = action[i] + anglist[i] 
                if newangle > -1.6 and newangle < 1.6:
                        servos[i].move_angle(newangle, blocking = False)
                
                        
        #if new move_angle angle is between -1.6 and 1.6, then move else, stay the same 
        '''
        s1.move_angle(action[0]+ ang1)
        s2.move_angle(action[1] + ang2)
        s3.move_angle(action[2]+ ang3)
        s4.move_angle(action[3]+ ang4)
        '''
        
        observedprime = (s1.read_angle(),s2.read_angle(),s3.read_angle(),s4.read_angle())
        #print(observedprime)
           
        return observedprime

def returnVal():
        return States, Actions
#print (States)

