import numpy as np
import math
#from simple_kanerva import *
import simple_kanerva


from agent_proj import *
from env_proj import *
import random
import sys, traceback





def returnKanerva():
    return state

def getStateActionvector(s,a):
    sa_vector = np.zeros(len(Actions) * numfeat)
    ind = Actions.index(a)
    #print("This is ind:", ind ,"numfeat", numfeat, "s", s)
    #sa_vector[ind  * numfeat:(ind + 1) * numfeat]
    sa_vector[s+ ind*(numfeat)] = 1
    #print("This is s", s)
    #print("state action vector",sa_vector)
    #print(sum(sa_vector))
    return sa_vector

def getProb(x):
    denom = float(1) / sum(
             [np.exp(np.dot(theta, getStateActionvector(x,b))) for b in Actions])         
    separate_probs = np.array(
        [np.exp(np.dot(theta,getStateActionvector(x,b)) ) for b in
         Actions]) * denom
    return separate_probs

def get_action(x):
    separate_probs = getProb(x)
    bins = [0]
    for i in range(len(Actions)):
        bins.append(bins[i] + separate_probs[i])
    #print("bins", bins)
    random_number = random.random()
    #print("randnum", random_number)
    index = np.digitize(random_number, bins)
    return Actions[index - 1]
    
def greedyPolicy(state):
    values = []
    for a in range (0,len(Actions)):
        x = getStateActionvector(state,Actions[a])
        v = x * weight
        #print("x", x, "w", weight, "v", v)
        values.append(v)
    index = numpy.argmax(values)
    return Actions[index]

def epPolicy(state):
    x = random.random()
    if x < epsilon:
        S,A = returnVal()
        action = random.choice(A)
    else:
        #find better way if more actions
        action =greedyPolicy(state)
    return action

def return_Q(state,action): 
    x = getStateActionvector(state,action)
    #print("x",x)
    #print("w",weight)
    v = np.dot(x,weight)
    #print("v", v)
    return v

def return_V(state): 
    v = sum(weight[state])
    return v



numSteps = 4000
numactfeat,numfeat = returnNums()
lamb = 0.4
alpha = 0.01/numactfeat #change in agent
epsilon = 0.1
theta = np.zeros(len(Actions) * numfeat)
smallweight = (0.1 * np.random.random(numfeat)) - 0.05 #^
weight = np.ones(numfeat) * 2
weight = weight + smallweight
States, Actions = returnVal() #change name 
time = 0
retval = 0
delta = 0




e_theta = np.zeros(len(Actions) * numfeat)
elig = np.zeros(numfeat) 
alpha_reward = 0.0002
alpha_theta = 0.1
reward_base = 0

kanerva_obj = KanervaCoding([-1.6,-1.6,-1.6,-1.6], [1.6,1.6,1.6,1.6], numfeat, random_seed = 26, bias = True)
s1,s2,s3,s4 = returnServos()
observations = init(s1,s2,s3,s4)
state = kanerva_obj.get_x(observations, numactfeat, ignore=None)



action = get_action(state)
sumzero = 0
countzero = 0
sumnonzero = 0
countnonzero = 0

while(time <= numSteps): #keyboard interrupt
    with open("abs_test_state_30.txt", "a") as f:
        #print("action", action)
        E = getProb(state) #init = 1 for seen 0 everything else
        #print("E", E)
        entropyE_orig = entropy(E)
        #print("entr_orig", entropyE_orig)
        #print("E", E) 
        observedprime = takeAction(action)#why do i even need to pass state? 
        #print("obsprime", observedprime)
        stateprime = kanerva_obj.get_x(observedprime, numactfeat, ignore=None, distance_metric='hamming')
        #print("stateprime", stateprime)
        #print("stateprime", stateprime)
   
        actionprime = get_action(stateprime)
        E_prime = getProb(stateprime) 
        entropyE_prime = entropy(E_prime)          
        
        reward = returnReward(entropyE_orig,entropyE_prime)
            
        #print("reward", reward)
        #reward_ave += reward/(time + 1)        
        
        if(action == (0,0,0,0)):
            countzero+=1
            sumzero += reward
        else:
            countnonzero+=1
            sumnonzero += reward
         
       
        # critic learns
        delta = reward - reward_base + return_V(stateprime) - return_V(state)
        #print("stateprime", stateprime, "state", state)
        #print("valuestateprime",return_V(stateprime), "valuestate", return_V(state))
        #print("delta", delta)
        reward_base += delta * alpha_reward 
        #print("reward base", reward_base)
        elig = lamb * elig 
        elig[state]= 1
        print("alpha", alpha, "delta", delta, "elig", elig)
        weight += alpha * delta * elig      
        
        #actor learns
        denom = float(1)/sum([np.exp(np.dot(theta, getStateActionvector(state,b))) for b in Actions])    
        statevector = np.zeros(numfeat)
        statevector[state] = 1
        separate_alla = [statevector *np.exp(np.dot(theta, getStateActionvector(state,b))) for b in Actions]     
        # xt's active feat are being scaled by e^preference of action taken --> just numerator 
        separate_probs = np.array([np.exp(np.dot(theta,getStateActionvector(state,b))) for b in Actions])     *denom
    
        state_alla = np.hstack(separate_alla)*denom 
       
        assert len(separate_probs) == len(Actions)
        rounded_at_index = random.choice(range(len(Actions)))
        cat = list(separate_probs[:rounded_at_index]) + list(separate_probs[rounded_at_index+1:])
        sumcat = sum(cat)
        rounded_prob = float(1) - float(sumcat)
        separate_probs = list(separate_probs[:rounded_at_index]) + list([rounded_prob]) + list(separate_probs[rounded_at_index+1:]) 
        
        #self.set_probs(separate_probs)
        #print("lamb", lamb, "eThet", e_theta.shape, "savect", len(getStateActionvector(state,action)), "state_alla", len(state_alla), "sepalla", len(separate_alla[0]))
        e_theta = lamb* e_theta + getStateActionvector(state,action)- state_alla #good = more likely, bad = less likely xtat???e_theta = self.lamda * self.e_theta + self.xtat - xt_alla
        theta += alpha_theta * delta * e_theta
    
    
        #print("etheta", e_theta)
        #print("theta", theta)
                 
        #print("E_prime", E_prime)
        
       
      
        
        
       
        #print( "reward",reward, "entropy", entropyE_prime, "action", action)
        print( "reward",reward, "entropy", entropyE_prime, "weight", weight)
        f.write(str(time)+','+str(reward)+ ',' + str(entropyE_prime) + ',' + str(E[0])+ ','+ str(E[1])+ ','+ str(E[2])+ ',' + str(return_V(state)) + '\n')
        #f.write(str(time) + ',' + str(return_V(state)) + '\n')
        #k = raw_input("stop")
        retval = retval + reward #return val: sum of all rewards in the episode
         
      
        state = stateprime
        action = actionprime 
        time += 1 
        #entropyE_orig = entropyE_prime
       

            
        
print("average return: ", retval/numSteps)
print("zeros", sumzero/countzero)
print("non zeros", sumnonzero/countnonzero)