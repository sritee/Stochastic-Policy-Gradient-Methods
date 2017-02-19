# softmax parametrization policy gradient using numerical gradient estimate
#we use num_rollouts episodes to estimate the gradient, using regression, as action selection is non determinstic
# A similar approach was used by UT Austin Robocup team to improve gait of a robot
#works well when we try to improve a base policy, rather than learn completely from scratch.
#sometimes might balance from start, restart the run as it was lucky initilization
import numpy as np
import gym
from numpy.linalg import inv

env=gym.make('CartPole-v1')
num_actions=2 #number of available actions,2 for cartpole,3 for acrobot and mountain car
num_episodes=2e3
fourier_order=2 #change order as desired, set to 0 to use raw observations
basealpha=1e-1 #change required base alpha
observations_dim=np.shape(env.observation_space.high)[0] #the observations in the environment
theta=np.random.normal(0,0.1,[pow(fourier_order+1,observations_dim),num_actions])
if fourier_order==0:
   theta=np.random.normal(0,0.1,[observations_dim,num_actions])
stepcount=np.zeros([num_episodes,1]) #count of steps per episode
visualize_after_steps=num_episodes-1 #start the display
num_rollouts=300 #how many training examples needed while regression, depends on fourier order
rewardacum=np.zeros([num_rollouts,2])
rewardthresh=400

def createalphas():  #different alpha for different order terms of fourier
    if fourier_order==0:
        return np.ones(np.shape(theta))*basealpha
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

def translate(value, leftMin, leftMax, rightMin, rightMax):
   
    leftrange = leftMax - leftMin
    rightrange = rightMax - rightMin
    #Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / leftrange
     #Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightrange)   
    
def normalize(state):
    
    normstate=np.empty(np.shape(state))
    val=np.array([2.5, 3.6, 0.28, 3.7]) #Upper lower limits
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

def evaluate(theta):
    rewardacum=0
    env.reset()
    curstate = env.observation_space.sample()
    curaction=pick_action(curstate,theta)  #epsilon greedy selection
    while True:
        #print(normalize(curstate))
        if i>visualize_after_steps:
            env.render()
        stepcount[i,0]=stepcount[i,0]+1
        nextstate,reward, done, info = env.step(curaction)                     
        rewardacum+=reward
        if done :
            return rewardacum
            break  
        if rewardacum>rewardthresh:
            print("Pole balanced succesfully")
            return rewardacum
            break
        nextaction=pick_action(nextstate,theta)
        curstate=nextstate
        curaction=nextaction
    
     
def pick_action(state,theta):
    state=computeFourierBasis(state)
    f=np.dot(theta.T,state)
    f=f-np.max(f) #for numerical stability
    denominator=  np.sum(np.exp(f))
    numerator=np.exp(f)
    val=numerator/denominator
    #if idx1==0:
        #print(np.max(val)*100)
    distribution=np.cumsum(val)# we have obtained the cumulative distribution
    temp=np.random.rand(1) 
    action=np.digitize(temp,distribution) #picking from the probability distro
    return int(action)

def compute_gradient(epsilonmatrix,reward_acum):
    y=np.expand_dims(reward_acum[:,0]-reward_acum[:,1],axis=1)
    x=np.transpose(epsilonmatrix).reshape(num_rollouts,-1)
    g= np.reshape(inv(x.T@x)@x.T@y,(-1,num_actions),order='F')/2 #symmetric gradient estimate by regression
    gclipped=g/np.linalg.norm(g)
    return gclipped
def update_actor(theta,epsilonmatrix,reward_acum):
    gradient=compute_gradient(epsilonmatrix,reward_acum)
    theta=theta+ alphavec*gradient
    return theta

for idx,i in enumerate(range(int(num_episodes))):
    epsilonmatrix=np.random.normal(0,5e-2,(theta.shape[0],theta.shape[1],num_rollouts))
    for idx1,k in enumerate(range(int(num_rollouts))):
        epsilon=epsilonmatrix[:,:,idx1]
        
        for idx2,j in enumerate(range(2)):
            if idx2==0:
                theta=theta+epsilon
            else:
                theta=theta-2*epsilon #2 to account for the fact that we added epsilon earlier
            curstate = env.reset()
            curaction=pick_action(curstate,theta)  #softmax
            step=0
            while True:
                if i>visualize_after_steps:
                    env.render()
                nextstate,reward, done, info = env.step(curaction)                 
                
                if done:
                    if idx2==1:
                        theta=theta+epsilon
                    break

                nextaction=pick_action(nextstate,theta)
                
        
                rewardacum[idx1,idx2]+=reward
                curstate=nextstate
                curaction=nextaction
                step+=1
                if step>rewardthresh:
                    print('Pole has Been Balanced!')
                    break
    num=evaluate(theta)
    print("episode %d finished after %d timesteps" %(idx,num))
    theta=update_actor(theta,epsilonmatrix,rewardacum)
    #print(theta[0])
env.monitor.close()

    
    
