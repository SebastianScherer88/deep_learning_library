# contains the dependencies, utils and main classes necessary to use diy deep learning library
#
# Utils/dependencies
# [0] basic imports
# [1] define some util functions
#
# Main layer classes
# [2] define a fully connected layer
# [3.1] define convolution layer utils
# [3.2] define convolution layer main class
# [4] define pooling layer main class
# [5] define fully connected to convolution reshaping layer class
# [6] define convolution to fully connected reshaping layer class
# [7] define dropout layer class
#
# Hidden input layer class
# [8] define 'secret' input layer class
#
# Network class
# [9] define feed forward network class

#----------------------------------------------------
# [0] Make some basic imports
#----------------------------------------------------

import numpy as np
from mnist_classes import get_mnist_data

#----------------------------------------------------
# [1] Define some computational util functions
#----------------------------------------------------

def tanh(Z):
    return np.tanh(Z)

def Dtanh(A):
    return 1 - np.multiply(A,A)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def Dsigmoid(A):
    return np.multiply(A,1 - A)

def relu(z,leak=0):
    assert (0 <= leak) and (leak < 0.5)
    return np.max(leak * Z,Z)

def Drelu(A):
    t1 = (A > 0) * 1
    t2 = (A <= 0) * leak
    return t1 + t2

def identity(Z):
    return Z

def Didentity(A):
    return np.ones(A.shape)

def softmax(Z):
    #print('From within softmax:')
    #print('Type of input array Z:',type(Z))
    return np.exp(Z) / np.sum(np.exp(Z),axis=1,keepdims=True)

def softmaxLoss(P,Y):
    return -np.mean(np.sum(np.multiply(Y,np.log(P)),axis=1))

def sigmoidLoss(P,Y):
    return -np.mean(np.sum(np.multiply(Y,np.log(P)) + np.multiply((1 - Y),np.log(1 - P)),axis=1))

#----------------------------------------------------
# [2] Fully connected layer class
#----------------------------------------------------

class FcLayer(object):
    '''Object class representing a fully connected layer in a feed-forward neural network'''
    def __init__(self,n,activation='tanh'):
        self.sizeIn = None
        self.sizeOut = [n]
        
        assert activation in ['tanh','sigmoid','relu','identity','softmax']
        if activation == 'tanh':
            self.activation = (tanh,activation)
            self.Dactivation = Dtanh
        elif activation == 'sigmoid': # possibly output layer, attach loss function just in case
            self.activation = (sigmoid,activation)
            self.Dactivation = Dsigmoid
        elif activation == 'relu':
            self.activation = (relu,activation)
            self.Dactivation = Drelu
        elif activation == 'identity':
            self.activation = (identity,activation)
            self.Dactivation = Didentity
        elif activation == 'softmax': # definitely output layer, so no need for Dactivation
            self.activation = (softmax,activation)
            
        self.Weight = None
        self.bias = None
        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
    def __str__(self):
        '''Returns a string with bio of layer.'''

        bio = '----------------------------------------' \
                + '\n Layer type: Fully connected layer' \
                + '\n Number of neurons in input data: ' + str(self.sizeIn) \
                + '\n Type of activation used: ' + self.activation[1] \
                + '\n Number of neurons in output data: ' + str(self.sizeOut) \
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        Z_c = np.dot(A_p,self.Weight) + self.bias
        #print("From within fully connected layer's forwardProp:")
        #print("Type of Z_c:", type(Z_c))
        A_c = self.activation[0](Z_c)
        #print("Type of A_c:", type(A_c))
        #print("From within fully connected layer's forwardProp:")
        #print("Shape of previous layer's activation:",A_p.shape)
        #print("Shape of current layer's activation:",A_c.shape)
        #print("------------------------------------")
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        A_c = self.cache['A']
        self.cache['DZ'] = np.multiply(self.Dactivation(A_c),DA_c)
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        A_p = self.previousLayer.cache['A']
        # calculate weight gradients
        DWeight = np.dot(A_p.T,DZ_c) / A_p.shape[0]
        Dbias = np.mean(DZ_c,axis=0)
        Dcache = {'DWeight':DWeight,'Dbias':Dbias}
        # calculate DZ_p, i.e. DZ of previous layer in network
        DA_p = np.dot(DZ_c,self.Weight.T)
        self.previousLayer.getDZ_c(DA_p)
        
        return Dcache
        
    def updateLayerParams(self,learningRate,Dcache):
        DWeight, Dbias = Dcache['DWeight'], Dcache['Dbias']
        self.Weight -= learningRate * DWeight
        self.bias -= learningRate * Dbias
        
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer
            
        self.initializeWeightBias()
        
    def initializeWeightBias(self):
        n_p,n_c = self.sizeIn[0],self.sizeOut[0]
        self.Weight = np.random.randn(n_p,n_c) * 1 / (n_p + n_c)
        self.bias = np.zeros((1,n_c))

#----------------------------------------------------
# # [3.1] define convolution layer utils
#----------------------------------------------------

def getPictureDims(height_pl,width_pl,padding_cl,kernelParams_cl):
    '''Calculates a convolutional layers height and width dimensions based on:
    - previous (convolutional) layer shape
    - type of padding used
    - kernel size of curent layer'''
    
    stride = kernelParams_cl['stride']
    if 'height_k' in kernelParams_cl: # dealing with a convolutional layer's request
        height_k, width_k = kernelParams_cl['height_k'],kernelParams_cl['width_k']
    elif 'height_pool' in kernelParams_cl: # dealing with a pooling layer's request
        height_k, width_k = kernelParams_cl['height_pool'],kernelParams_cl['width_pool']
    
    if padding_cl == 'valid':
        height_pad, width_pad = (0,0)
    if padding_cl == 'same':
        height_pad = np.ceil((stride*(height_pl-1)+height_k-height_pl) / 2)
        width_pad = np.ceil((stride*(width_pl-1)+width_k-width_pl) / 2)
        
    height_cl = int(np.ceil((height_pl-height_k+1+2*height_pad) / stride))
    width_cl = int(np.ceil((width_pl-width_k+1+2*width_pad) / stride))
    
    return height_cl, width_cl

def getConvSliceCorners(h,w,height_k,width_k,stride):
    '''Calculates and returns the edge indices of the slice in layer_p used to compute layer_c[h,w]'''
    hStart, hEnd = (h * stride, h * stride + height_k)
    wStart, wEnd = (w * stride, w * stride + width_k)
    
    return hStart,hEnd,wStart,wEnd

def pad(Z,pad):
    # Takes a four-dimensionan tensor tensor of shape (x1,x2,x3,x4,x5)
    # Adds zero padding for dimensions x2 and x3 to create an array
    # Zpadded of shape (x1,x2+2*pad,x3*pad,x4)
    Zpadded = np.pad(Z,mode='constant',
                     pad_width=((0,0),(0,0),(pad,pad),(pad,pad),(0,0)),
                    constant_values=((0,0),(0,0),(0,0),(0,0),(0,0)))
    
    return Zpadded

#----------------------------------------------------
# [3.2] define convolution layer main class
#----------------------------------------------------

class ConvLayer(object):
    '''Object class representing a convolutional layer in a feed-forward neural net'''
    
    def __init__(self,kernelHeight,kernelWidth,channels,
                 stride,padding='valid',
                 activation='tanh'):
    
        assert padding in ['same','valid']
        assert activation in ['tanh','sigmoid','relu','identity']
        
        if activation == 'tanh':
            self.activation = (tanh,activation)
            self.Dactivation = Dtanh
        elif activation == 'sigmoid':
            self.activation = (sigmoid,activation)
            self.Dactivation = Dsigmoid
        elif activation == 'relu':
            self.activation = (relu,activation)
            self.Dactivation = Drelu
        elif activation == 'identity':
            self.activation = (identity,activation)
            self.Dactivation = Didentity
            
        self.padding = padding
        self.sizeIn = None
        self.sizeOut = [channels]
        self.kernelParams = {'stride':stride,'height_k':kernelHeight,'width_k':kernelWidth}
        
        self.Weight = None
        self.bias = None
        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        
        bio = '----------------------------------------' \
                + '\n Layer type: Convolution layer' \
                + '\n Shape of kernel (width,height): ' + ','.join([str(self.kernelParams['height_k']),
                                                                  str(self.kernelParams['width_k'])]) \
                + '\n Stride used for kernel: ' + str(self.kernelParams['stride']) \
                + '\n Shape of input data (channels,height,width): ' + str(self.sizeIn) \
                + '\n Padding used: ' + self.padding \
                + '\n Type of activation used: ' + self.activation[1] \
                + '\n Shape of output data (channels,height,width): ' + str(self.sizeOut)
            
        return bio

        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_k = self.kernelParams['height_k']
        width_k = self.kernelParams['width_k']
        stride = self.kernelParams['stride']
        
        Z_c = np.zeros((batchSize,channels_c,height_c,width_c,1))
        
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_k,width_k,stride)
                X_hw = np.multiply(A_p[:,:,hStart:hEnd,wStart:wEnd,:],self.Weight)
                Y_hw = np.sum(X_hw,axis=(1,2,3))
                Z_c[:,:,h,w,0] = Y_hw
        
        Z_c += self.bias
        A_c = self.activation[0](Z_c)
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        A_c = self.cache['A']
        self.cache['DZ'] = self.Dactivation(A_c) * DA_c
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_k = self.kernelParams['height_k']
        width_k = self.kernelParams['width_k']
        stride = self.kernelParams['stride']
        
        channels_p = self.sizeIn[0]
        height_p = self.sizeIn[1]
        width_p = self.sizeIn[2]
        
        # calculate weight gradients & DZ_p, i.e. DZ of previous layer in network
        DWeight = np.zeros((1,channels_p,height_k,width_k,channels_c))
        DZ_cback = np.transpose(DZ_c,(0,4,2,3,1))
        
        DA_p = np.zeros((batchSize,height_p,width_p,channels_p,1))
        Weight_back = np.transpose(self.Weight,(0,4,2,3,1))
        
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_k,width_k,stride)
                #print('A')
                I_hw = np.multiply(DZ_cback[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:],
                                   A_p[:,:,hStart:hEnd,wStart:wEnd,:])

                J_hw = np.mean(I_hw,axis=0)

                DWeight[0,:,:,:,:] += J_hw
                
                X_hw = np.multiply(DZ_c[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:],
                                   Weight_back)
                #print('E')
                Y_hw = np.sum(X_hw,axis=1)

                DA_p[:,hStart:hEnd,wStart:wEnd,:,0] += Y_hw
                
        #DWeight /= DA_p.shape[0]
        
        Dbias = np.mean(np.sum(DZ_c,axis=(2,3,4)),axis=0)
        Dbias = Dbias[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        
        Dcache = {'DWeight':DWeight,'Dbias':Dbias}
        
        DA_p = np.transpose(DA_p,(0,3,1,2,4))
        self.previousLayer.getDZ_c(DA_p)
        
        return Dcache
    
    def updateLayerParams(self,learningRate,Dcache):
        DWeight, Dbias = Dcache['DWeight'], Dcache['Dbias']
        self.Weight -= learningRate * DWeight
        self.bias -= learningRate * Dbias
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        height_pl, width_pl = self.sizeIn[1],self.sizeIn[2]
        height_cl, width_cl = getPictureDims(height_pl,
                                           width_pl,
                                           self.padding,
                                           self.kernelParams)
        
        self.sizeOut.extend([height_cl,width_cl])
        
        self.nextLayer = nextLayer
            
        self.initializeWeightBias()
        
    def initializeWeightBias(self):
        self.Weight = np.random.randn(1,
                                      self.sizeIn[0], # channels_pl
                                      self.kernelParams['height_k'], # kernel height
                                      self.kernelParams['width_k'], # kernel width
                                      self.sizeOut[0]) # channels_cl
        self.bias = np.zeros((1,self.sizeOut[0],1,1,1)) # channels_cl

#----------------------------------------------------
# [4] define pooling layer main class
#----------------------------------------------------

class PoolingLayer(object):
    '''Pooling layer (either "max" or "mean") between convolutional/appropriate reshaping layers'''
    
    def __init__(self,poolingHeight,poolingWidth,
                 stride,padding='valid',
                 poolingType='max'):
        
        assert padding in ['same','valid']
        assert poolingType in ['max','mean']
        
        self.padding = padding
        self.poolingType = poolingType
        self.sizeIn = None
        self.sizeOut = None
        self.poolingParams = {'stride':stride,'height_pool':poolingHeight,'width_pool':poolingWidth}
        
        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        
        bio = '----------------------------------------' \
                + '\n Layer type: Pooling layer (' + str(self.poolingType) +')' \
                + '\n Shape of pool (width,height): ' + ','.join([str(self.poolingParams['height_pool']),
                                                                  str(self.poolingParams['width_pool'])]) \
                + '\n Stride used for pool: ' + str(self.poolingParams['stride']) \
                + '\n Shape of input data (channels,height,width): ' + str(self.sizeIn) \
                + '\n Padding used: ' + self.padding \
                + '\n Shape of output data (channels,height,width): ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_pool = self.poolingParams['height_pool']
        width_pool = self.poolingParams['width_pool']
        stride = self.poolingParams['stride']
        
        Z_c = np.zeros((batchSize,channels_c,height_c,width_c,1))
                
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_pool,width_pool,stride)
                X_hw = A_p[:,:,hStart:hEnd,wStart:wEnd,0]
                
                if self.poolingType == 'max':
                    Y_hw = np.amax(X_hw,axis=(2,3))
                elif self.poolingType == 'mean':
                    Y_hw = np.mean(X_hw,axis=(2,3))
                    
                Z_c[:,:,h,w,0] = Y_hw
        
        self.cache['A'] = Z_c # pooling layer has no activation, hence A_c = Z_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        self.cache['DZ'] = DA_c # since for pooling layers A = Z -> DA = DZ
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_pool = self.poolingParams['height_pool']
        width_pool = self.poolingParams['width_pool']
        stride = self.poolingParams['stride']
        
        channels_p = self.sizeIn[0]
        height_p = self.sizeIn[1]
        width_p = self.sizeIn[2]

        # calculate weight gradients & DZ_p, i.e. DZ of previous layer in network
        
        DA_p = np.zeros((batchSize,channels_p,height_p,width_p,1))
        
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_pool,width_pool,stride)
                
                if self.poolingType == 'max':
                    X_hw = A_p[:,:,hStart:hEnd,wStart:wEnd,0]
                    #print("From within pooling layer's backprop:")
                    #print("Shape of X_hw:",X_hw.shape)
                    Y_hw = np.amax(X_hw,(2,3))[:,:,np.newaxis,np.newaxis]
                    #print("Shape of Y_hw:",Y_hw.shape)
                    U_hw = DZ_c[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:]
                    #print('Shape of U_hw:',U_hw.shape)
                    V_hw = (X_hw==Y_hw)[:,:,:,:,np.newaxis]
                    #print('Shape of V_hw:',V_hw.shape)
                    #print('Shape of updated DA_p slice:',DA_p[:,:,hStart:hEnd,wStart:wEnd,:].shape)
                    DA_p[:,:,hStart:hEnd,wStart:wEnd,:] += U_hw * V_hw
                elif self.poolingType == 'mean':
                    X_hw = 1/(height_pool * width_pool) * np.ones(batchSize,channels_p,height_pool,width_pool,1)
                    DA_p[:,:,hStart:hEnd,wStart:wEnd,:] += np.multiply(DZ_c[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:],
                                                                       X_hw)
        
        self.previousLayer.getDZ_c(DA_p)
        
        return
        
    def updateLayerParams(self,learningRate,Dcache):
        """Bogus class to make neural network's backprop more homogenuous"""
        return
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.sizeOut = [self.sizeIn[0]] # number of channels remains unchanged through pooling
        height_pl, width_pl = self.sizeIn[1],self.sizeIn[2]
        height_cl, width_cl = getPictureDims(height_pl,
                                           width_pl,
                                           self.padding,
                                           self.poolingParams)
        
        self.sizeOut.extend([height_cl,width_cl])
        
        self.nextLayer = nextLayer

#----------------------------------------------------
# [5] define fully connected to convolution reshaping layer class
#----------------------------------------------------

class FcToConv(object):
    '''Transitional layer handling reshaping between fclayer activations -> convLayer activations,
    and convLayer activation derivatives -> fcLayer activation derivatives.'''
    
    def __init__(self,convDims):

        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
        self.sizeIn = None
        
        [convChannels,convHeight,convWidth] = convDims
        self.sizeOut = [convChannels,convHeight,convWidth]
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        bio = '----------------------------------------' \
                + '\n Layer type: Reshaping layer (Fully connected -> Convolution)' \
                + '\n Shape of input data: ' + str(self.sizeIn) \
                + '\n Shape of output data: ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        aShape = [batchSize,self.sizeOut[0],sizeOut[1],sizeOut[2],1]
        A_c = Z_c = A_p.reshape(aShape)
        #print("From within reshape (conv -> fc) layer's forwardProp:")
        #print("Shape of previous layer's activation:",A_p.shape)
        #print("Shape of current layer's activation:",A_c.shape)
        #print("------------------------------------")
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        batchSize = DA_c.shape[0]
        dzShape = [batchSize, self.sizeIn[0]]
        self.cache['DZ'] = DA_c.reshape(dzShape)
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        # calculate DZ_p, i.e. DZ of previous layer in network
        DA_p = DZ_c
        self.previousLayer.getDZ_c(DA_p)
        
        return
    
    def updateLayerParams(self,learningRate,Dcache):
        # bogus function for layer consistency from the neural net class point of view
        return
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer
        self.convShape = self.nextLayer.convShape

#----------------------------------------------------
# [6] define convolution to fully connected reshaping layer class
#----------------------------------------------------

class ConvToFC(object):
    '''Transitional layer handling reshaping between convlayer activations -> fcLayer activations,
    and fcLayer activation derivatives -> convLayer activation derivatives.'''
    
    def __init__(self,n):

        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
        self.sizeIn = None
        self.sizeOut = [n]
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        bio = '----------------------------------------' \
                + '\n Layer type: Reshaping layer (Convolution -> Fully connected)' \
                + '\n Shape of input data: ' + str(self.sizeIn) \
                + '\n Shape of output data: ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        #A_c = A_p.reshape([batchSize].extend(self.sizeOut))
        A_c = A_p.reshape([batchSize,self.sizeOut[0]])
        #print("From within reshape (conv -> fc) layer's forwardProp:")
        #print("Shape of previous layer's activation:",A_p.shape)
        #print("Shape of current layer's activation:",A_c.shape)
        #print("------------------------------------")
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        batchSize = DA_c.shape[0]
        #dzShape = [batchSize].extend(self.sizeIn).extend(1)
        dzShape = [batchSize,self.sizeIn[0],self.sizeIn[1],self.sizeIn[2],1]
        self.cache['DZ'] = DA_c.reshape(dzShape)
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        # calculate DZ_p, i.e. DZ of previous layer in network
        DA_p = DZ_c
        self.previousLayer.getDZ_c(DA_p)
        
        return
    
    def updateLayerParams(self,learningRate,Dcache):
        # bogus function for layer consistency from the neural net class point of view
        return
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer

#----------------------------------------------------
# [7] define dropout layer class
#----------------------------------------------------

class Dropout(object):
    '''Dropout layer acting between "real" network layers.'''
    
    def __init__(self,dropoutRate):
        self.dropoutRate = dropoutRate
        self.cache = {'A':None,'DZ':None,'outDropper':None}
        self.previousLayer = None
        self.nextLayer = None
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        bio = '----------------------------------------' \
                + '\n Layer type: Dropout layer' \
                + '\n Dropout rate: ' + str(self.dropoutRate) \
                + '\n Shape of input/output data: ' + str(self.sizeOut)
            
        return bio
        
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        outDropper = np.random.choice([0,1],size=A_p.shape,p=[self.dropoutRate,1-self.dropoutRate])
        A_c = Z_c = np.multiply(outDropper,A_p)
        self.cache['A'] = A_c
        self.cache['outDropper'] = outDropper
        
    def getDZ_c(self,DA_c):
        DZ_c = DA_c
        self.cache['DZ'] = DZ_c
        
    def backwardProp(self):
        DZ_c = self.cache['DZ']
        outDropper = self.cache['outDropper']
        DA_p = np.multiply(DZ_c,outDropper)
        self.previousLayer.getDZ_c(DA_p)
        
        return
        
    def updateLayerParams(self,learningRate,Dcache):
        # bogus function for layer consistency from the neural net class point of view
        return
        
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.sizeOut = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer

#----------------------------------------------------
# [8] define 'secret' input layer
#----------------------------------------------------

class _InputLayer(object):
    '''Input layer created automatically by network once the training data shape/kind is known to it.'''
    
    def __init__(self,pad=0):
        self.sizeOut = None
        self.cache = {'A':None}
        self.nextLayer = None
        self.flatData = None
        self.pad = pad
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        
        if self.flatData:
            secondLine = '\n Number of neurons in input/output data: ' + str(self.sizeOut)
        elif not self.flatData:
            secondLine = '\n Shape of input/output data (channels,height,width): ' + str(self.sizeOut)
        
        bio = '----------------------------------------' \
                + '\n Layer type: (Secret) input layer' \
                + secondLine
            
        return bio
        
    def forwardProp(self,XBatch):
        if self.flatData:
            self.cache['A'] = XBatch # -> input data is 2 dimensional, (bachSize,nFeature)
            #print("From within secret input layer's forward prop:")
            #print("Type of input batch without adding bogus dimension:",type(self.cache['A']))
        elif not self.flatData:
            self.cache['A'] = pad(np.expand_dims(XBatch,-1),self.pad) # -> input data is 4 dim.,(sampleSize,channels,height,width)
            #print("From within secret input layer's forward prop:")
            #print("Shape of input batch after adding bogus dimension:",self.cache['A'].shape)
            
        #print("-------------------------------")
        
        return
    
    def getDZ_c(self,DA_c):
        # bogus function for layer consistency from the neural net class point of view
        return
    
    def makeReady(self,nextLayer=None,XSample=None):
        self.nextLayer = nextLayer
        self.sizeOut = self.getSizeOutFromX(XSample)
        
    def getSizeOutFromX(self,XSample):
        if len(XSample.shape) == 2: # -> assume X is flattened array of shape (sampleSize,nFeature)
            sizeOut = [XSample.shape[1]]
            
            self.flatData = True
            
            return sizeOut
        
        elif len(XSample.shape) == 4: # -> assume X is high dim. tensor of shape (sampleSize,channels,height,width)
            inputChannels,inputHeight,inputWidth = XSample.shape[1:]
            sizeOut = [inputChannels,inputHeight+2*self.pad,inputWidth+2*self.pad]
            
            self.flatData = False
            
            return sizeOut
        
        else:
            print('''X has to be either of shape [nSamples,nFeatures] or, for images,
            [imageChannels,imageHeight,imageWidth]. Please reshape your training data and
            try compiling the model again.''')

#----------------------------------------------------
# [9] define feed forward network class
#----------------------------------------------------

class FFNetwork(object):
    
    def __init__(self,initPad=0):
        self.initPad = initPad
        self.layers = []
        self.loss = None
        self._inputLayer = None
        self.dataType = None # will indicate wether flattened feature vectors or high dim image tensors
        self.finalState = False
        self.trained = False
        
    def __str__(self):
        '''Print out structure of neural net (if it has been fixated).'''
        
        if self.finalState:
            bluePrint = '\n'.join([self._inputLayer.__str__()] + [layer.__str__() for layer in self.layers])
            
            return bluePrint
        else:
            print('The model has to be fixated first.')
        
    def addFCLayer(self,n,activation='tanh'):
        '''Adds a fully connected layer to the neural network.'''
        
        fullyConnectedLayer = FcLayer(n,activation)
        
        self.layers.append(fullyConnectedLayer)
        
    def addConvLayer(self,kernelHeight,kernelWidth,channels,stride,padding='valid',activation='tanh'):
        '''Adds a convolution layer to the neural network.'''
        
        convolutionLayer = ConvLayer(kernelHeight,
                                     kernelWidth,
                                     channels,
                                     stride,
                                     padding,activation)
        
        self.layers.append(convolutionLayer)
        
    def addPoolingLayer(self,poolingHeight,poolingWidth,stride,padding='valid',poolingType='max'):
        '''Adds a pooling layer to the neural network. Recommended after convolutional layers.'''
        
        poolingLayer = PoolingLayer(poolingHeight,
                                    poolingWidth,
                                    stride,
                                    padding,
                                    poolingType)
        
        self.layers.append(poolingLayer)

        
    def addFCToConvReshapeLayer(self,convDims):
        '''Adds a reshaping layer to the neural network. Necessary to link up a fully connected layer
        with a subsequent convolution layer.'''
        
        shapeFullyConnectedToConvolution = FcToConv(convDims)
        
        self.layers.append(shapeFullyConnectedToConvolution)
        
    def addConvToFCReshapeLayer(self,n):
        '''Adds a reshaping layer to the neural network. Necessary to link up a convolutional layer with a 
        subsequent fully connected layer.'''
        
        shapeConvolutionalToFullyConnected = ConvToFC(n)
        
        self.layers.append(shapeConvolutionalToFullyConnected)
        
    def addDropoutLayer(self,dropoutRate):
        '''Adds a dropout layer.'''
        
        dropoutLayer = Dropout(dropoutRate)
        
        self.layers.append(dropoutLayer)
        
    def fixateNetwork(self,XSample):
        '''Fixes model, finalising its blue-print.
        Attaches loss function to model.
        Creates hidden input layer based on shape of passed sample.
        Calls each layer's makeReady() method.'''
        
        # only do stuff if model hasnt allready been fixated
        if self.finalState:
            print('This model has already been fixated.')
            
            return
        
        # add secret input layer and make ready
        self._inputLayer = _InputLayer(self.initPad)
        self._inputLayer.makeReady(self.layers[0],XSample)
        
        # iterate through layers and introduce to immediate neighouring layers
        for i, layer in enumerate(self.layers):
            if i == 0: # first layer, use _inputLayer as previous layer
                previousLayer = self._inputLayer
            else: # at least second layer, so pass previous layer
                previousLayer = self.layers[i-1]
            
            if i == len(self.layers) - 1: # last user made layer in network, no next layer exists
                nextLayer = None
            else: # at most second to last layer, pass next layer
                nextLayer = self.layers[i+1]
                
            layer.makeReady(previousLayer,nextLayer)
        
        lastLayer = self.layers[-1]
        
        # attach loss function to neural net depending on last fully connected layer's activation type
        if lastLayer.activation[0] == sigmoid:
            self.loss = sigmoidLoss
        elif lastLayer.activation[0] == softmax:
            self.loss = softmaxLoss
        else:
            print('The last layer needs to have either "softmax" or "sigmoid" activation. Model was not fixated')
        
        self.finalState = True
        
    def trainNetwork(self,nEpochs,learningRate,batchSize,X,y,displaySteps=50,oneHotY = True):
        '''Trains the neural network using naive gradient descent.'''
        # vectorize Y to one-hot format if needed (default is True)
        if oneHotY:
            Y = self.oneHotY(y)
        elif not oneHotY:
            Y = y
        
        # initialize storage for batch losses to be collected during training
        lossHistory = []
        recentLoss = 0
        
        # execute training
        for epoch in range(nEpochs):
            for i,(XBatch,YBatch) in enumerate(self.getBatches(X,Y,batchSize)):
                P = self.forwardProp(XBatch)
                batchLoss = self.loss(P,YBatch)
                recentLoss += batchLoss
                self.backwardProp(learningRate,YBatch)
                
                if (i % displayStep) == 0 and (i != 0):
                    averageRecentLoss = recentLoss / displaySteps
                    lossHistory.append(averageRecentLoss)
                    recentLoss = 0
                    print('Epoch', epoch)
                    print('Batch', i)
                    print('Loss averaged over last '+str(displaySteps)+' batches',averageRecentLoss)
        
        # announce end of training
        self.trained = True
        print('---------------------------------------------------')
        print('Training finished.')
        print('nEpochs:',nEpochs)
        print('learningRate:',learningRate)
        print('batchSize:',batchSize)
        
        return lossHistory
        
    def oneHotY(self,y):
        '''One hot vectorizes a target class index list into a [nData,nClasses] array.'''
        
        nClasses = len(np.unique(y.reshape(-1)))
        Y = np.eye(nClasses)[y.reshape(-1)]
        
        return Y
    
    def getBatches(self,X,Y,batchSize):
        '''Sample randomly from X and Y, then yield batches.'''
        nData = X.shape[0]
        shuffledIndices = np.arange(nData)
        np.random.shuffle(shuffledIndices)
        
        XShuffled, YShuffled = X[shuffledIndices], Y[shuffledIndices]
        
        nBatches = int(X.shape[0] / batchSize)
        
        for iBatch in range(nBatches):
            XBatch, YBatch = (XShuffled[iBatch*batchSize:(iBatch+1)*batchSize],
                              YShuffled[iBatch*batchSize:(iBatch+1)*batchSize])
        
            yield XBatch, YBatch
        
    def forwardProp(self,XBatch):
        '''Executes one forward propagation through the network. Returns the loss averaged over the batch.'''
        
        self._inputLayer.forwardProp(XBatch) # feed training batch into network
        
        for layer in self.layers: # forward propagate through the network
            layer.forwardProp()
            
        P = self.layers[-1].cache['A'] # get prediction
            
        return P
        
    def backwardProp(self,learningRate,YBatch):
        '''Executes one backward propagation through the network. Returns the network's parameter's gradients'''
        
        P = self.layers[-1].cache['A']
        self.layers[-1].cache['DZ'] = (P - YBatch) #/ YBatch.shape[0]
        
        for i,layer in enumerate(reversed(self.layers)):
            layerDCache = layer.backwardProp()
            layer.updateLayerParams(learningRate,layerDCache)
            
    def predict(self,X):
        '''If model is trained, performs forward prop and returns the prediction array.'''
        
        if not self.trained:
            print('Model needs to be trained first.')
            
            return
        
        P = self.forwardProp(X)
        
        return np.argmax(P,axis=1).reshape(-1,1)