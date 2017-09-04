#agent has a learning ???? 
from simple_kanerva import * 
from env_proj import *


States,Actions = returnVal()
numfeat = 50
numactfeat = 15
E = [0]*len(Actions)
#print("len of Actions", len(Actions))
#initialize where 1 is seen and 0 is everything else; E is same length as numfeat

allfeat = range(numfeat)


def retProb(t,oc,probability_o):
        '''
        if t==0:
                probability = 1
        else:
        '''
        if t !=0:
                probability = probability_o + (1.0/t)*(oc - probability_o) #occurred = 1 if happened, 0 if did not happen
        else:
                probability = oc
             
        
        return probability

#prob of each feature being active 


def updateE(action_set, t):
        #print("E before", E)
        print("action set", action_set)
        for i in range(len(Actions)):
                #print("E[i]",E[i])
                if Actions[i] == action_set:
                        E[i] = retProb(t,1,E[i])
                else:
                        
                        E[i] = retProb(t,0,E[i])                                                
        #print("E after: ",E)
        return E
#print("results",updateE((0,0,0,0), 0))
'''
def UpdateE(obsd_state,t):
        #print("in the update e method; this is len of states", len(States))
        for i in range(len(States)):
                #print("i",i)
                if (States[i] == obsd_state).all(): #(A==B).all():
                        print("if E[i] before",E[i])
                        E[i] = retProb(t,1,E[i]) 
                        print("if E[i] after",E[i])
                else:
                        print("else E[i] before",E[i])
                        E[i] = retProb(t,0,E[i])
                        print("else E[i] after",E[i])
                        
        print("E", E)
        return E
'''
'''
kanerva_obj = KanervaCoding([-1.6,-1.6,-1.6,-1.6], [1.6,1.6,1.6,1.6], numfeat, random_seed = 26, bias = True)
s1,s2,s3,s4 = returnServos()
observations = init(s1,s2,s3,s4)
state = kanerva_obj.get_x(observations, numactfeat, ignore=None)

def returnKanerva():
        return state
  '''  
def returnNums():
        return numactfeat,numfeat


#UpdateE((42,12,3,22, 4), 0)

