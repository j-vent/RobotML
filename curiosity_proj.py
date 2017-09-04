#IGM
import math
#import scipy

constant = 1

def entropy(E):#scipy.stats.entropy(E)
    H = 0
    for x in (E):
        #print ("x", x)
        if(x > 0):
            H -= x * math.log(x,2) #base 2
      
    #print("entropy:", H)
    #print(H)
    return H

def returnReward(E_o, E_p):
    reward = constant * abs(E_o - E_p) 
    return reward 

'''
def better_entropy(E):
    #H = scipy.stats.entropy(E,base= 10)
    print (H)
    return H

constant = 1
E_orig= [4.0/9, 2.0/9, 3.0/9] #probability at different times
E_prime = [1.0/9, 2.0/9, 6.0/9] 

#print(entropy(E, 3, 0))
reward = constant * (entropy(E_orig) - entropy(E_prime))
print("reward",reward)
#reward = constant * (better_entropy(E) - better_entropy(E))
#print("reward",reward)
#print(better_entropy(E))


'''