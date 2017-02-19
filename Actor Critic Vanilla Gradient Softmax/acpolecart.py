#Actor Critic method using softmax parametrization
import numpy as np
import gym

env=gym.make('CartPole-v1')
num_actions=2#number of available actions
num_episodes=500
fourier_order=3#change order as desired, set to 0 to use raw observations
basealpha=1e-2
observations_dim=np.shape(env.observation_space.high)[0] #the observations in the environment
rewardthresh=300 # succcesful if it balances for 300 steps

if fourier_order==0:
   w=np.zeros([observations_dim,num_actions])
theta=np.random.normal(0,0.1,[pow(fourier_order+1,observations_dim),num_actions])
if fourier_order==0:
   theta=np.random.rand(observations_dim,num_actions) 
w=np.random.normal(0,0.1,[pow(fourier_order+1,observations_dim),num_actions])
stepcount=np.zeros([num_episodes,1])
gamma=0.99 #discount factor
zeta=0.9#bootstrapping parameter, note that lambda is keyword in python
visualize_after_steps=num_episodes-5 #visualize the last 5 runs

def createalphas():  #different alpha for different order terms of fourier
    if fourier_order==0:
        return np.ones([observations_dim,num_actions])*basealpha
    temp=tuple([np.arange(fourier_order+1)]*observations_dim)
    b=np.array(np.meshgrid(*temp)).T.reshape(-1,observations_dim)
    c=np.linalg.norm(b,axis=1)
    d=basealpha/c
    d[0]=basealpha
    d = np.expand_dims(d, axis=1)
    alphavec=np.tile(d,num_actions)
    alphavec=np.reshape(alphavec,(-1,num_actions))
    return alphavec

alphavec=createalphas() #create the learning rate matrix
betavec=alphavec*1

def translate(value, leftMin, leftMax, rightMin, rightMax):
   
    leftrange = leftMax - leftMin
    rightrange = rightMax - rightMin
    #Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / leftrange
     #Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightrange)   
    
def normalize(state):
    
    normstate=np.empty(np.shape(state))
    val1=np.array([2.5, 3.6, 0.28, 3.7]) #in case you want to set manually the limits
    val=-val1
    for i in range(np.shape(state)[0]):
        normstate[i]=translate(state[i],val[i],val1[i],0,1)
    #print (normstate)
    return normstate
    
def computeFourierBasis(state):
    
    if fourier_order==0:  #no cosine terms at all
        return normalize(state)
    normstate=normalize(state)
    temp=tuple([np.arange(fourier_order+1)]*observations_dim)
    b=np.array(np.meshgrid(*temp)).T.reshape(-1,observations_dim)
    return np.cos(np.pi*np.dot(b,normstate))    

def computevalue(w,state,action): #compute value of taking some state in some state
    
    statefeature=computeFourierBasis(curstate)
    statefeaturetemp=np.tile(statefeature[:,np.newaxis],[1,num_actions])
    f=np.dot(theta.T,statefeature)
    f=f-np.max(f) #numerical stability
    denominator=np.sum(np.exp(f))
    numerator=np.exp(f)
    softmax=numerator/denominator
    statefeaturetemp=np.tile(statefeature[:,np.newaxis],[1,num_actions])
    temp=-softmax*statefeaturetemp
    temp[:,action]+=statefeature
    x=np.dot(np.ravel(temp,order='F'),np.ravel(w,order='F'))
    return x
    
 
def pick_action(state,theta):
    state=computeFourierBasis(state)
    f=np.dot(theta.T,state)
    f=f-np.max(f) #for numerical stability
    denominator=  np.sum(np.exp(f))
    numerator=np.exp(f)
    val=numerator/denominator
    #print(np.max(val)*100)
    distribution=np.cumsum(val)# we have obtained the cumulative distribution
    
    temp=np.random.rand(1) 
    action=np.digitize(temp,distribution) #picking from the probability distro'''
    return int(action)
    
  
def update_critic(w,delta,e):
    
    w= w+ delta*betavec*e
    return w
    
def update_actor(theta,w,curstate,action):
    
    statefeature=computeFourierBasis(curstate)
    statefeaturetemp=np.tile(statefeature[:,np.newaxis],[1,num_actions])
    f=np.dot(theta.T,statefeature)
    f=f-np.max(f) #numerical stability
    denominator=np.sum(np.exp(f))#denominator computed once to save time
    numerator=np.exp(f)
    softmax=numerator/denominator
    qval=computevalue(w,curstate,action)
    theta-= alphavec*statefeaturetemp*softmax*qval
    theta[:,action]+= statefeature*alphavec[:,0]*qval #correcting the derivative term
    return theta

#env.monitor.start('/tmp/acrobot-experiment-1',force='True') #recommended off, there seems to minor bug where code doesnt enter the last if condition if this is on

for i in range(int(num_episodes)):
    curstate = env.reset()
    curaction=pick_action(curstate,theta)  #epsilon greedy selection
    e=np.zeros(np.shape(w))
    while True:
        #print(normalize(curstate))
        if i>visualize_after_steps:
            env.render()
        stepcount[i,0]=stepcount[i,0]+1
        e[:,curaction]=e[:,curaction]+ computeFourierBasis(curstate); #accumulating traces
        nextstate,reward, done, info = env.step(curaction) 
        delta = reward - computevalue(w,curstate,curaction);   #The TD Error                    

        if done:
            print("Episode %d finished after %d timesteps" %(i,stepcount[i,0]))
            theta=update_actor(theta,w,curstate,curaction)
            w=update_critic(w,delta,e)
            #print (theta[0])
        
            break
        
        nextaction=pick_action(nextstate,theta)
        #print(nextaction)
        
        delta=delta+ gamma*computevalue(w,nextstate,nextaction)
        #print(delta)
        theta=update_actor(theta,w,curstate,curaction)
        w=update_critic(w,delta,e)
        
        curstate=nextstate
        curaction=nextaction
        e=e*gamma*zeta
        
        if stepcount[i,0]>rewardthresh:
            print('Pole has been balanced')
            alphavec=alphavec*0.9
            betavec=betavec*0.9
            break
        #print(np.max(theta))
env.monitor.close()

    
    
