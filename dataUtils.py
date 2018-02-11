# 161041024
# Ahmet Soyyiğit

from os import listdir
from os import chdir
from os import getcwd
from os.path import join

import numpy as np

class Joint:
    HipCenter = 6
    Spine = 3
    ShoulderCenter = 2
    Head = 19
    ShoulderLeft = 1
    ElbowLeft = 8
    WristLeft = 10
    HandLeft = 12
    ShoulderRight = 0
    ElbowRight = 7
    WristRight = 9
    HandRight = 11
    HipLeft = 5
    KneeLeft = 14
    AnkleLeft = 16
    FootLeft = 18
    HipRight = 4
    KneeRight = 13
    AnkleRight = 15
    FootRight = 17

edgesBetweenJoints = (
    (Joint.ShoulderCenter,Joint.ShoulderLeft),
    (Joint.ShoulderLeft, Joint.ElbowLeft),
    (Joint.ElbowLeft, Joint.WristLeft),
    (Joint.WristLeft, Joint.HandLeft),

    (Joint.ShoulderCenter,Joint.ShoulderRight),
    (Joint.ShoulderRight,Joint.ElbowRight),
    (Joint.ElbowRight,Joint.WristRight),
    (Joint.WristRight,Joint.HandRight),

    (Joint.Head,Joint.ShoulderCenter),
    (Joint.ShoulderCenter, Joint.Spine),
    (Joint.Spine, Joint.HipCenter),

    (Joint.HipCenter, Joint.HipRight),
    (Joint.HipRight, Joint.KneeRight),
    (Joint.KneeRight, Joint.AnkleRight),
    (Joint.AnkleRight, Joint.FootRight),

    (Joint.HipCenter, Joint.HipLeft),
    (Joint.HipLeft, Joint.KneeLeft),
    (Joint.KneeLeft, Joint.AnkleLeft),
    (Joint.AnkleLeft, Joint.FootLeft),
    )

# This function can read test, train and validation data, you need to call it for each of them seper.
def readMSRDataset(data_dir,timesteps,normalize=True):
    print('Loading MSR 3D Data, data directory %s' % data_dir)
    numOfJoints = 20
    maxValue = 3.879377  # dataset attribute 
    minValue = -1.878035 # dataset attribute
    frameSeqs = 25000 # first, allocate more than needed, after reading, delete unnecessary allocation

    prevDir = getcwd()
    chdir(data_dir)
    documents = [d for d in sorted(listdir("."))]
    
    inpData = np.zeros((timesteps,frameSeqs,numOfJoints*3), dtype=np.float32)
    labels = np.zeros((frameSeqs), dtype=np.int64)
    batchLens = np.zeros((len(documents),2), dtype=np.int64)
    trainPrevActionIdx = 0
    for fIdx,file in enumerate(documents):
        currentLabel = int(file[1:3]) 
        action = np.loadtxt(file)
        action = np.delete(action, 3, axis=1) # delete the unnecessary last column
        numOfFrames = action.shape[0] // numOfJoints
        action = np.reshape(action,( numOfFrames, numOfJoints*3 ))
        if normalize:
            # Frame normalization
            spineCoordinates = action[:,Joint.Spine*3:Joint.Spine*3+3]
            hipCenterCoordinates = action[:,Joint.HipCenter*3:Joint.HipCenter*3+3]
            for i in range(0,60,3):
                action[:,i:i+3] -= (spineCoordinates+hipCenterCoordinates)/2
            # Dataset normalization
            action=(action+abs(minValue))/(maxValue+abs(minValue))

        zeroPaddedAction = np.concatenate((np.zeros((timesteps-1, numOfJoints*3)), action))
        batchLens[fIdx] = [trainPrevActionIdx, numOfFrames-1+trainPrevActionIdx]
        bs, be = batchLens[fIdx]
        
        for step in range(timesteps):
            inpData[step,bs:be] = zeroPaddedAction[step:step+numOfFrames-1]
        labels[bs:be] = currentLabel-1
        trainPrevActionIdx = numOfFrames-1+trainPrevActionIdx

    # free unnecessary allocation
    inpData = np.delete(inpData, range(batchLens[-1][1],frameSeqs) , axis=1)
    labels = np.delete(labels, range(batchLens[-1][1],frameSeqs), axis=0)
    chdir(prevDir)

    return inpData,labels

def readMSRDatasetCrossSubject(data_dir,timesteps,subjectToTest,diff=False, normalize=True):
    if subjectToTest < 1 or subjectToTest > 10:
        print("Error, invalid subject number")
        return None, None
    print('Loading MSR 3D Data, data directory %s' % data_dir)
    numOfJoints = 20
    maxValue = 3.879377  # dataset attribute 
    minValue = -1.878035 # dataset attribute
    frameSeqs = 120000 # first, allocate more than needed, after reading, delete unnecessary allocation

    prevDir = getcwd()
    chdir(data_dir)
    documents = [d for d in sorted(listdir("."))]
    
    trainData = np.zeros((timesteps,frameSeqs,numOfJoints*3), dtype=np.float32)
    trainLabels = np.zeros((frameSeqs), dtype=np.int64)
    testData = np.zeros((timesteps,frameSeqs//5,numOfJoints*3), dtype=np.float32)
    testLabels = np.zeros((frameSeqs//5), dtype=np.int64)
    trainPrevActionIdx = 0
    testPrevActionIdx = 0
    for fIdx,file in enumerate(documents):
        currentLabel = int(file[1:3])
        currentSubject = int(file[5:7])
        action = np.loadtxt(file)
        action = np.delete(action, 3, axis=1) # delete the unnecessary last column
        numOfFrames = action.shape[0] // numOfJoints
        action = np.reshape(action,( numOfFrames, numOfJoints*3 ))
        if normalize:
            # Frame normalization
            spineCoordinates = action[:,Joint.Spine*3:Joint.Spine*3+3]
            hipCenterCoordinates = action[:,Joint.HipCenter*3:Joint.HipCenter*3+3]
            for i in range(0,60,3):
                action[:,i:i+3] -= (spineCoordinates+hipCenterCoordinates)/2
            # Dataset normalization
            action=(action+abs(minValue))/(maxValue+abs(minValue))

        zeroPaddedAction = np.concatenate((np.zeros((timesteps-1, numOfJoints*3)), action))
        if diff:
##            diffFactor = 0.0166
            for i in range(timesteps-1,zeroPaddedAction.shape[0]):
                zeroPaddedAction[i] -= zeroPaddedAction[i-1]
##                zeroPaddedAction[i] = ((numOfJoints*3-1)-np.argsort(zeroPaddedAction[i])) * diffFactor            
        
        if currentSubject == subjectToTest:
            bs, be = testPrevActionIdx , numOfFrames-1+testPrevActionIdx
            for step in range(timesteps):
                testData[step,bs:be] = zeroPaddedAction[step:step+numOfFrames-1]
            testLabels[bs:be] = currentLabel-1
            testPrevActionIdx += numOfFrames-1
        else:
            # I will also use dataset augmentation here, and make data triple more!
            differences = (zeroPaddedAction[1:] - zeroPaddedAction[:-1]) / 2
            differences = np.concatenate((differences, np.zeros((1, numOfJoints*3))))
            for i in range(2):
                bs, be = trainPrevActionIdx , trainPrevActionIdx+numOfFrames-1            
                for step in range(timesteps):
                    trainData[step,bs:be] = zeroPaddedAction[step:step+numOfFrames-1]
                trainLabels[bs:be] = currentLabel-1
                trainPrevActionIdx += numOfFrames-1
                zeroPaddedAction += differences
    # free unnecessary allocation
    trainData   = np.delete(trainData, range(trainPrevActionIdx,frameSeqs) , axis=1)
    trainLabels = np.delete(trainLabels, range(trainPrevActionIdx,frameSeqs), axis=0)
    testData    = np.delete(testData, range(testPrevActionIdx,frameSeqs//5) , axis=1)
    testLabels  = np.delete(testLabels, range(testPrevActionIdx,frameSeqs//5), axis=0)
    chdir(prevDir)

    return trainData,trainLabels,testData,testLabels

def organizeSkeletonData(data):
    # These numbers are written according to Joint enumeration of Kinect
    llData = data[:,:,[15,16,17,42,43,44,48,49,50,54,55,56]]
    rlData = data[:,:,[12,13,14,39,40,41,45,46,47,51,52,53]]
    laData = data[:,:,[36,37,38,30,31,32,24,25,26,3,4,5]]
    raData = data[:,:,[33,34,35,27,28,29,21,22,23,0,1,2]]
    sData  = data[:,:,[57,58,59,6,7,8,9,10,11,18,19,20]]
    return np.concatenate((llData,laData,sData,raData,rlData),2)

##for i in range(1,2):
##    a,b,c,d = readMSRDatasetCrossSubject("./MSRAction3DSkeletonReal3D",40,i)
##    a = organizeSkeletonData(a)
##    c = organizeSkeletonData(c)
##    print(a.shape)
##    print(b.shape)
##    print(c.shape)
##    print(d.shape)


