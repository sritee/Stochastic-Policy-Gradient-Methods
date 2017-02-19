#Reinforce with and  without Baseline
# This uses simple baseline as average of rewards returned so far to reduce variance
# Refer paper Policy gradient Methods in Robotics for the optimal baseline
# Despite baseline,as with all monte carlo methods, suffers from variance, depending on task horizon length
import numpy as np
import gym
baselineused='on' #Baseline is on
env=gym.make('CartPole-v1')
num_actions=2
num_episodes=3e3
fourier_order=3 #change order as desired, set to 0 to use raw observations
basealpha=1e-4#change required base alpha if baseline is not set
if baselineused=='on':
    basealpha=basealpha*1
observations_dim=np.shape(env.observation_space.high)[0] #the observations in the environment
theta=np.random.normal(0,0.05 ,[pow(fourier_order+1,observations_dim),num_actions]) #non uniform initialiaztion centred at zero
if fourier_order==0:
   theta=np.random.normal(0,0.05,observations_dim,num_actions) 
stepcount=np.zeros([num_episodes,1])
gamma=0.98#discount factor
visualize_after_steps=num_episodes #start the display
valhistory=np.array([0])
wincount=0
rewardthresh=300 #succesfull if it balances for 300 steps!
def createalphas(basealpha):  #different alpha for different order terms of fourier
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


def translate(value, leftMin, leftMax, rightMin, rightMax):
   
    leftrange = leftMax - leftMin
    rightrange = rightMax - rightMin
    #Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / leftrange
     #Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightrange)   
    
def normalize(state):
    
    normstate=np.empty(np.shape(state))

    val=np.array([2.5, 3.6, 0.28, 3.7]) #in case you want to set manually the limits
    val1=-val
    
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
 
 
def pick_action(state,theta):
    state=computeFourierBasis(state)
    f=np.dot(theta.T,state)
    f=f-np.max(f) #for numerical stability
    denominator=  np.sum(np.exp(f))
    numerator=np.exp(f)
    val=numerator/denominator
    #print(np.max(val)*100)
    distribution=np.cumsum(val)# we have obtained the cumulative distribution
    #print(distribution)
    temp=np.random.rand(1) 
    action=np.digitize(temp,distribution) #picking from the probability distro'''
    #print(action)
    return int(action)
    
def update_actor(theta,trajectory,rewardhistory,actionhistory,baseline):
    trajectory=np.reshape(trajectory,[-1,observations_dim])
    temp=np.zeros(np.shape(theta))
    if baselineused=='off':
        baseline=0
    for k in range(trajectory.shape[0]):
    
        
        discounting=np.array([pow(gamma,j) for j in np.arange(trajectory[k:-1,:].shape[0])])
        rewardacum=rewardhistory[k:-1]
        mcreturn=np.dot(rewardacum,discounting)
        statefeature=computeFourierBasis(trajectory[k,:])
        statefeaturetemp=np.tile(statefeature[:,np.newaxis],[1,num_actions])
        f=np.dot(theta.T,statefeature)
        f=f-np.max(f) #numerical stability
        denominator=np.sum(np.exp(f))#denominator computed once to save time
        numerator=np.exp(f)
        softmax=numerator/denominator
        temp-= alphavec*statefeaturetemp*softmax*(mcreturn-baseline)
        temp[:,actionhistory[k]]+= statefeature*alphavec[:,0]*(mcreturn-baseline)#correcting the derivative te
    #print(np.max(np.abs(temp)))
    return theta+temp

#env.monitor.start('/tmp/cartpole-experiment-1',force='True') #switch on for visualiztion
alphavec=createalphas(basealpha) #create the learning rate matrix
for idx,i in enumerate(range(int(num_episodes))):
    curstate = env.reset()
    curaction=pick_action(curstate,theta)  #softmax selection
    rewardhistory=np.array([])
    trajectory=np.array([])
    actionhistory=np.array([])
    rewardacum=0
    while True:
        #print(normalize(curstate))
        if i>visualize_after_steps:
            env.render()
        stepcount[i,0]=stepcount[i,0]+1
        nextstate,reward, done, info = env.step(curaction) 
        if done:
            print("Episode %d finished after %d timesteps" %(i,stepcount[i,0]))
            trajectory=np.append(trajectory,curstate)
            rewardhistory=np.append(rewardhistory,reward)
            actionhistory=np.append(actionhistory,curaction)
            break
        
        nextaction=pick_action(nextstate,theta)
        #print(nextaction)
        rewardacum+=reward
        trajectory=np.append(trajectory,curstate)
        rewardhistory=np.append(rewardhistory,reward)
        actionhistory=np.append(actionhistory,curaction)
        curstate=nextstate
        curaction=nextaction
        
        if stepcount[i,0]>rewardthresh:
            print('Pole Balanced succesfully!')
            wincount+=1#alpha reduction
            if (np.sum(stepcount[i-30:-1,0])/30)>rewardthresh:
                alphavec=0.5*alphavec
            break
      
    baseline=np.sum(valhistory)/(idx+1)
    valhistory=np.append(valhistory,rewardacum)
    theta=update_actor(theta,trajectory,rewardhistory,actionhistory,baseline)
    
env.monitor.close()

    
    
    
