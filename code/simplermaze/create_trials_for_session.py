#code to generate trials based on phase:
import pandas as pd
import numpy as np
#import random

numTrials = 300
sessionStage = 2 #values between 1 and 4


gratingMap = pd.read_csv("grating_maps.csv")
rewardSequences = pd.read_csv("reward_sequences.csv")

stage = rewardSequences[rewardSequences.sessionID=="Stage "+str(sessionStage)]
#lenRewLoc = len(stage.rewloc)

trialsDistribution = dict()
for index,location in enumerate(stage.rewloc):
    subTrials = int(np.floor(numTrials*list(stage.portprob)[index]))
    probRewardLocation = np.random.choice([1,0],numTrials,p=[list(stage.rewprob)[index],
                                                            1-list(stage.rewprob)[index]])
    
    trialTuples = list()
    
        
    for i in range(subTrials):
        trialTuples.append((location,probRewardLocation[i],list(stage.wrongallowed)[index]))
    
    trialsDistribution[location] = trialTuples

allTogether=list()

for item in trialsDistribution.keys():
    allTogether+=(trialsDistribution[item])#this keeps the list flat
                                           #(as opposed to a list of lists)

#now shuffle the list
np.random.shuffle(allTogether)

# create DataFrame using data
trials = pd.DataFrame(allTogether, columns =['location', 'givereward', 'wrongallowed'])

