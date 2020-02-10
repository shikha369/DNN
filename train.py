#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:16:33 2017

"""
import argparse
import pickle
import numpy as np

### IMPORTANT NOTATIONS
'''
  num_datapoints = num_examples
  batch= batch_size
  num_epochs = num_passes
  num_batches = iterations
  
'''

FLAGS = argparse.ArgumentParser(
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
FLAGS.add_argument('-lr', '--lr',type=float,
                 help='input learning rate')
FLAGS.add_argument('-mom', '--momentum',type=float,
                 help='input momentum')
FLAGS.add_argument('-num_hidden', '--num_hidden',type=int,
                 help='No. of hidden layers')
FLAGS.add_argument('-sizes', '--sizes',
                 help='size of hidden layers')
FLAGS.add_argument('-activation', '--activation',
                 help='tanh/sigmoid')
FLAGS.add_argument('-loss', '--loss',
                 help='sq/ce')
FLAGS.add_argument('-opt', '--opt',
                 help='gd/momentum/nag/adam')
FLAGS.add_argument('-batch_size', '--batch_size',type=int,
                 help='batch size..valid values are 1 and multiples of 5')
FLAGS.add_argument('-anneal', '--anneal',
                 help='True/False')
FLAGS.add_argument('-save_model_path', '--save_dir',
                 help='path to save your model')
FLAGS.add_argument('-log_path', '--expt_dir',
                 help='path to save your logs')
FLAGS.add_argument('-data_path', '--mnist',metavar='FILE',
                 help='path to load data')

X=None
y=None
num_datapoints=None ##total number of data points
valid_set_x= None
valid_set_y=None
test_set_x=None
test_set_y=None

def onehotEncoding(batch_true,batch_size):
    Truelabels = np.zeros((batch_size, 10))
    for i in range(batch_size):
             Truelabels[i,batch_true[i]] = 1
    return Truelabels
 
def predict(model_data):
    X = model_data['X']
    choice=model_data['activation'] 
    nh = model_data['nh']

    # Forward propagation
           
    a={}                 ##  Activation for each layer in NN
           
    #Input layer
    z1 = X.dot(model_data['W1']) + model_data['B1'] # Preactivation
    a1 = activation(z1,choice)                      # activation
    a.update({'A1':a1})
           
    #Hidden layers
    for itr_l in range(nh-1):
        z = a['A'+str(itr_l+1)].dot(model_data['W'+str(itr_l+2)]) + model_data['B'+str(itr_l+2)]
        act = activation(z,choice)  
        a.update({'A'+str(itr_l+2):act})
               
    #Output layers    
    zL = a['A'+str(nh)].dot(model_data['W'+str(nh+1)]) + model_data['B'+str(nh+1)]
    exp_scores = np.exp(zL)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs #return Softmax values
    
def calculate_loss(model_data):
    y, num_datapoints = model_data['y'],model_data['num_datapoints']
    loss=model_data['loss']
    # Forward propagation to calculate our predictions
    softmax_prob = predict(model_data)
    predicted=np.argmax(softmax_prob,axis=1)
    #write file predicted here
    #print model_data['if_printPredictions']
    if model_data['if_printPredictions'] :

      np.savetxt(model_data['p_path'], predicted,fmt='%i', delimiter='\n')
    #print predicted
    correct=(predicted==y)
    count=0.0
    for i in xrange(0,num_datapoints):
      if correct[i] : count=count +1
               
    # Calculating the loss
    
    if loss=='ce':
      true_logprobs = -np.log(softmax_prob[range(num_datapoints), y])  
      data_loss = np.sum(true_logprobs)
    elif loss=='sq':
      predicted_out  = softmax_prob  ## softmax values..
      predicted_out[range(num_datapoints),y] -= 1 
      predicted_out = np.square(predicted_out)
      data_loss=np.sum(predicted_out)
    else:
        print 'from predict else'
    return data_loss/num_datapoints, (count*100)/num_datapoints  
    #Return loss and ACCURACY
    #return data_loss/num_datapoints, count


def activation(z,choice):
    if choice=='tanh':
      return np.tanh(z)
    elif choice=='sigmoid':
      return 1 / (1 + np.exp(-z))
    else : 
      print 'from activation else'

def derivative(a,choice):
    if choice=='tanh':
      return (1 - np.power(a, 2))
    if choice=='sigmoid':
      return (a - np.power(a, 2))
    else : 
      print 'from derivative else'

''' Testing.... <Todo>
    def calculate_grad_sq(probs,deltaL,batch):
        grad_l_y = (-2)*deltaL

        c1 = (np.multiply(grad_l_y[:,0] , (probs[:,0]-np.square(probs[:,0])))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,0] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,0] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,0] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,0] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,0] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,0] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,0] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,0] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,0] ,probs[:,9])))
        
        c1 = c1.reshape((batch,1))
          
        c2 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,1] ,probs[:,0]))
        + np.multiply(grad_l_y[:,1] , (probs[:,1]-np.square(probs[:,1])))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,1] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,1] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,1] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,1] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,1] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,1] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,1] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,1] ,probs[:,9])))
        
        c2 = c2.reshape((batch,1))

        c3 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,2] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,2] ,probs[:,1]))
        + np.multiply(grad_l_y[:,2] , (probs[:,2]-np.square(probs[:,2])))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,2] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,2] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,2] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,2] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,2] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,2] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,2] ,probs[:,9])))
        
        c3 = c3.reshape((batch,1))
          
        c4 =( - np.multiply(grad_l_y[:,0] , np.multiply(probs[:,3] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,3] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,3] ,probs[:,2]))
        + np.multiply(grad_l_y[:,3] , (probs[:,3]-np.square(probs[:,3])))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,3] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,3] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,3] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,3] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,3] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,3] ,probs[:,9])))
        
        c4 = c4.reshape((batch,1))
        
        c5 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,4] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,4] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,4] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,4] ,probs[:,2]))
        + np.multiply(grad_l_y[:,4] , (probs[:,4]-np.square(probs[:,4])))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,4] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,4] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,4] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,4] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,4] ,probs[:,9])))
        
        c5 = c5.reshape((batch,1))
        
        c6 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,5] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,5] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,5] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,5] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,5] ,probs[:,4]))
        + np.multiply(grad_l_y[:,5] ,(probs[:,5]-np.square(probs[:,5])))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,5] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,5] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,5] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,5] ,probs[:,9])))
        
        c6 = c6.reshape((batch,1))
        
        c7 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,6] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,6] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,6] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,6] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,6] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,6] ,probs[:,5]))
        + np.multiply(grad_l_y[:,6] , (probs[:,6]-np.square(probs[:,6])))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,6] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,6] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,6] ,probs[:,9])))
        
        c7 = c7.reshape((batch,1))
        
        c8 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,7] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,7] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,7] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,7] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,7] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,7] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,7] ,probs[:,6]))
        + np.multiply(grad_l_y[:,7] , (probs[:,7]-np.square(probs[:,7])))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,7] ,probs[:,8]))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,7] ,probs[:,9])))
        
        c8 = c8.reshape((batch,1))

        c9 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,8] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,8] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,8] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,8] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,8] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,8] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,8] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,8] ,probs[:,7]))
        + np.multiply(grad_l_y[:,8] , (probs[:,8]-np.square(probs[:,8])))
        - np.multiply(grad_l_y[:,9] , np.multiply(probs[:,8] ,probs[:,9])))
        
        c9 = c9.reshape((batch,1))
        
        c10 = (- np.multiply(grad_l_y[:,0] , np.multiply(probs[:,9] ,probs[:,0]))
        - np.multiply(grad_l_y[:,1] , np.multiply(probs[:,9] ,probs[:,1]))
        - np.multiply(grad_l_y[:,2] , np.multiply(probs[:,9] ,probs[:,2]))
        - np.multiply(grad_l_y[:,3] , np.multiply(probs[:,9] ,probs[:,3]))
        - np.multiply(grad_l_y[:,4] , np.multiply(probs[:,9] ,probs[:,4]))
        - np.multiply(grad_l_y[:,5] , np.multiply(probs[:,9] ,probs[:,5]))
        - np.multiply(grad_l_y[:,6] , np.multiply(probs[:,9] ,probs[:,6]))
        - np.multiply(grad_l_y[:,7] , np.multiply(probs[:,9] ,probs[:,7]))
        - np.multiply(grad_l_y[:,8] , np.multiply(probs[:,9] ,probs[:,8]))
        + np.multiply(grad_l_y[:,9] , (probs[:,9]-np.square(probs[:,9]))))
        
        c10 = c10.reshape((batch,1))


        deltaL = np.hstack((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10))
        #print deltaL.shape
        deltaL = np.asarray(deltaL)
                
        return deltaL
'''
    
def build_model(args):
    
    datasets = pickle.load(open(args.mnist,"rb"))
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    X=train_set_x
    y=train_set_y

    num_datapoints = len(X) # training set size HAND PICKED HERE.....
    nn_input_dim = len(X[0]) # input layer dimensionality
    nn_output_dim = len(np.unique(y)) # output layer dimensionality
    
    num_epochs=20 #Number of Epochs
    count_Zero_training_loss_annealed=0
    
    batch = args.batch_size
    loss_=args.loss
    choice=args.activation
    print loss_
    print choice
    if_anneal=args.anneal
    
    folt = open(args.expt_dir+"log_loss_train.txt", "w")
    foat = open(args.expt_dir+"log_error_train.txt", "w")
    folv = open(args.expt_dir+"log_loss_valid.txt", "w")
    foav = open(args.expt_dir+"log_error_valid.txt", "w")
    fols = open(args.expt_dir+"log_loss_test.txt", "w")
    foas = open(args.expt_dir+"log_error_test.txt", "w")
    
    test_predictions_path = args.expt_dir+"test_predictions.txt"
    valid_predictions_path  = args.expt_dir+"valid_predictions.txt"
    
    num_batches = num_datapoints / batch
    if num_datapoints % batch != 0:
      num_batches += 1
    
    epsilon = args.lr
    momentum = args.momentum
    opt = args.opt
    
    nh=args.num_hidden
    print nh
    
    np.random.seed(1)
    l=args.sizes.split(',')
    nn_hdim = np.zeros(nh)
    for i in range(nh):
      nn_hdim[i]=int(l[i])

    
    
    model_W = {}
    model_w_backup = {}
    prev_update_backup = {}
    first_o_m_backup = {}
    second_o_v_backup = {}
    
    nn_hdim = nn_hdim.astype(int)
    ## initialise for input layer ::
    W1=2*np.random.randn(nn_input_dim, nn_hdim[0])/np.sqrt(nn_input_dim +nn_hdim[0])
    b1 = np.zeros((1, nn_hdim[0]))
    dict_temp = {'W1':W1,'B1':b1}
    model_W.update(dict_temp)
    
    ##initialise for Hidden layers excluding input and output
    for i_l in range(nh-1):
        W = 2*np.random.randn(int(nn_hdim[i_l]) ,int(nn_hdim[i_l+1]))/np.sqrt(nn_hdim[i_l] +nn_hdim[i_l+1])
        b = np.zeros((1, int(nn_hdim[i_l+1])))
        dict_temp = {'W'+str(i_l+2):W,'B'+str(i_l+2):b}
        model_W.update(dict_temp)
        
    ## initialise for outer layer ::
    W = 2*np.random.randn(int(nn_hdim[nh-1]) ,nn_output_dim)/np.sqrt(nn_hdim[nh-1] + nn_output_dim)
    b = np.zeros((1, nn_output_dim))
    dict_temp = {'W'+str(nh+1):W,'B'+str(nh+1):b}
    model_W.update(dict_temp)    
        
    #dynamic previous updates
    
    prev_update = {}
    prev_update_w1 = np.zeros((nn_input_dim, nn_hdim[0]))
    prev_update_b1 = np.zeros((1, nn_hdim[0]))
    dict_temp = {'prev_update_w1':prev_update_w1,'prev_update_b1':prev_update_b1}
    prev_update.update(dict_temp)
    
    for i_l in range(nh-1):
        prev_update_w = np.zeros((nn_hdim[i_l], nn_hdim[i_l+1]))
        prev_update_b = np.zeros((1, nn_hdim[i_l+1]))
        dict_temp = {'prev_update_w'+str(i_l+2):prev_update_w,'prev_update_b'+str(i_l+2):prev_update_b}
        prev_update.update(dict_temp)
    
    prev_update_w = np.zeros((nn_hdim[nh-1],nn_output_dim))  
    prev_update_b = np.zeros((1,nn_output_dim))
    dict_temp  = {'prev_update_w'+str(nh+1):prev_update_w,'prev_update_b'+str(nh+1):prev_update_b}
    prev_update.update(dict_temp)
    
    # This is what model we return at the end
    model = {}
    
    prev_validation_accuracy = 0
    
    # initialisations specific to Adam
    
    t = 0;
    first_o_m = {}               # To store first order moments
    second_o_v = {}              # To store Second order moments
               
    m_w1 = np.zeros((nn_input_dim, nn_hdim[0]))
    v_w1 = m_w1
    m_b1 = np.zeros((1, nn_hdim[0]))
    v_b1 = m_b1
    dict_temp_m = {'m_w1':m_w1,'m_b1':m_b1}
    dict_temp_v = {'v_w1':v_w1,'v_b1':v_b1}
    first_o_m.update(dict_temp_m)
    second_o_v.update(dict_temp_v)
    
    for i_l in range(nh-1):
        m_w_temp = np.zeros((nn_hdim[i_l], nn_hdim[i_l+1]))
        v_w_temp = m_w_temp
        m_b_temp = np.zeros((1, nn_hdim[i_l+1]))
        v_b_temp = m_b_temp
        dict_temp_m = {'m_w'+str(i_l+2):m_w_temp,'m_b'+str(i_l+2):m_b_temp}
        dict_temp_v = {'v_w'+str(i_l+2):v_w_temp,'v_b'+str(i_l+2):v_b_temp}                                  
        first_o_m.update(dict_temp_m)
        second_o_v.update(dict_temp_v)
    
    m_w = np.zeros((nn_hdim[nh-1],nn_output_dim))  
    v_w = m_w
    m_b = np.zeros((1,nn_output_dim))
    v_b = m_b
    dict_temp_m  = {'m_w'+str(nh+1):m_w,'m_b'+str(nh+1):m_b}
    dict_temp_v  = {'v_w'+str(nh+1):v_w,'v_b'+str(nh+1):v_b}              
    first_o_m.update(dict_temp_m)
    second_o_v.update(dict_temp_v)
    
    beta_1,beta_2 = 0.9,0.999        ## handpicked
    smoothing_factor = 0.00000001    ## Handpicked
    
    
    
    
    
    
    
    for itr in xrange(0, num_epochs):
    # code to reshuffle the dataset here...............
       p = np.random.permutation(len(X))
       X = X[p]
       y = y[p]
       step=0;
       print "epoch "+str(itr)
       for j in xrange(0, num_batches):
           ##First batch ...
           batch_X, batch_y = X[j * batch : (j + 1) * batch], y[j * batch : (j + 1) * batch]
                                
           # Forward propagation
           
           a={}                                            ##  Activation for each layer in NN
           
           #Input layer
           z1 = batch_X.dot(model_W['W1']) + model_W['B1'] # Preactivation
           a1 = activation(z1,choice)                      # activation
           a.update({'A1':a1})
           
           #Hidden layers
           for itr_l in range(nh-1):
               z = a['A'+str(itr_l+1)].dot(model_W['W'+str(itr_l+2)]) + model_W['B'+str(itr_l+2)]
               act = activation(z,choice)  
               a.update({'A'+str(itr_l+2):act})
               
           #Output layers    
           zL = a['A'+str(nh)].dot(model_W['W'+str(nh+1)]) + model_W['B'+str(nh+1)]
           exp_scores = np.exp(zL)
           probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
           # Backpropagation
           
           delta={}       #temp var to maintain delta (current) 
           deltaL = probs ## same size as probs
           
           # check for loss
           if loss_=='ce' :
                deltaL[range(batch), batch_y] -= 1
           if loss_=='sq' :
                Truelabels = onehotEncoding(batch_y,batch)
                deltaL[range(batch), batch_y] -= 1

                deltaL = calculate_grad_sq(probs,deltaL,batch)
           
           #Dynamic gradient calculation:
           dict_temp = {'delta':deltaL}
           delta.update(dict_temp)  
           
           grad_W= {}        # DATA structure for storing gradients:
               
           #Gradient for Outer layer::
           dW_out = ((a['A'+str(nh)]).T).dot(deltaL)
           db_out = np.sum(deltaL,axis = 0,keepdims = True)
           dict_temp = {'dw'+str(nh+1):dW_out,'db'+str(nh+1):db_out}
           grad_W.update(dict_temp)
           
           #Gradient for Hidden layers :
           for i_l in range(nh-1):
               delta_temp = ( delta['delta'].dot( model_W['W'+str(nh+1-i_l)].T ) )*derivative(a['A'+str(nh-i_l)] ,choice)
               dw = np.dot(a['A'+str(nh-i_l-1)].T,delta_temp)
               db = np.sum(delta_temp,axis = 0)
               delta.update({'delta':delta_temp})
               #Add weights to dw DATA STRUCTURE
               dict_temp = {'dw'+str(nh-i_l):dw,'db'+str(nh-i_l):db}
               grad_W.update(dict_temp)
           
           #gradients for Input layer :
           delta_i  = (delta['delta'].dot(model_W['W2'].T)) * derivative(a['A1'],choice)
           dwi = np.dot(batch_X.T,delta_i)
           dbi = np.sum(delta_i,axis = 0)
           dict_temp = {'dw1':dwi,'db1':dbi}
           grad_W.update(dict_temp)
           
 
         # Optimisation algorithm  parameter update
           
           # Dynamically declare update
           
           update_ = {}
           if opt=='gd' :
               
            for i in range(nh+1):
                update_temp_w = -epsilon * grad_W['dw'+str(i+1)] /batch
                update_temp_b = -epsilon * grad_W['db'+str(i+1)] /batch                                                  
                update_.update({'update_W'+str(i+1):update_temp_w})
                update_.update({'update_b'+str(i+1):update_temp_b})

           
          # momentum parameter upadate
           
           elif opt=='momentum' :
            
            for i in range(nh+1):
                update_temp_w = -epsilon * grad_W['dw'+str(i+1)] /batch + momentum * prev_update['prev_update_w'+str(i+1)]
                update_temp_b = -epsilon * grad_W['db'+str(i+1)] /batch + momentum * prev_update['prev_update_b'+str(i+1)]                                               
                update_.update({'update_W'+str(i+1):update_temp_w})
                update_.update({'update_b'+str(i+1):update_temp_b})   
           
           elif opt == 'nag' :
            # partial updates :
                
                # x_ahead = x + mu * prev_update
                
                # DYNAMIC PARTIAL UPDATE
                partial_update = {}
                for i in range(nh+1):
                    partial_update_w = momentum*prev_update['prev_update_w'+str(i+1)]
                    partial_update_b = momentum*prev_update['prev_update_b'+str(i+1)]
                    partial_update.update({'partial_update_w'+str(i+1):partial_update_w , 'partial_update_b'+str(i+1):partial_update_b})                   
                
                
                # DYNAMIC LOOKAHEADS:
                
                w_ahead={}
                for i in range(nh+1):
                    w_new = model_W['W'+str(i+1)] + partial_update['partial_update_w'+str(i+1)]
                    b_new = model_W['B'+str(i+1)] + partial_update['partial_update_b'+str(i+1)]    
                    w_ahead.update({'W'+str(i+1)+'_ahead':w_new,'B'+str(i+1)+'_ahead':b_new})   
                
                
                #calculate gradients at lookahead 
                
                ## code repeated...
                
                # DYNAMIC METHOD 
                #Feedforward
                
                a={}                 ##  Activation for each layer in NN
           
                #Input layer
                z1 = batch_X.dot(w_ahead['W1_ahead']) + w_ahead['B1_ahead'] # Preactivation
                a1 = activation(z1,choice)                                  # activation
                a.update({'A1':a1})
           
                #Hidden layers
                for itr_l in range(nh-1):
                   z = a['A'+str(itr_l+1)].dot(w_ahead['W'+str(itr_l+2)+'_ahead']) + w_ahead['B'+str(itr_l+2)+'_ahead']
                   act = activation(z,choice)  
                   a.update({'A'+str(itr_l+2):act})
               
                #Output layers    
                zL = a['A'+str(nh)].dot(w_ahead['W'+str(nh+1)+'_ahead']) + w_ahead['B'+str(nh+1)+'_ahead']
                exp_scores = np.exp(zL)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
           
                # Backpropagation
           
                delta={}       #temp var to maintain delta (current) 
                deltaL = probs ## same size as probs
           
                # check for loss
                if loss_=='ce' :
                    deltaL[range(batch), batch_y] -= 1
                if loss_=='sq' :
                    Truelabels = onehotEncoding(batch_y,batch)
                    deltaL[range(batch), batch_y] -= 1

                    deltaL = calculate_grad_sq(probs,deltaL,batch)
           
                #Dynamic gradient calculation:
                dict_temp = {'delta':deltaL}
                delta.update(dict_temp)  
           
                grad_W_ahead= {} # DATA structure for storing gradients:
               
                #Gradient for Outer layer::
                dW_out = ((a['A'+str(nh)]).T).dot(deltaL)
                db_out = np.sum(deltaL,axis = 0,keepdims = True)
                dict_temp = {'dw'+str(nh+1):dW_out,'db'+str(nh+1):db_out}
                grad_W_ahead.update(dict_temp)
                          
                #Gradient for Hidden layers :
                for i_l in range(nh-1):
                  delta_temp = ( delta['delta'].dot( w_ahead['W'+str(nh+1-i_l)+'_ahead'].T ) )*derivative(a['A'+str(nh-i_l)] ,choice)
                  dw = np.dot((a['A'+str(nh-i_l-1)].T),delta_temp)
                  db = np.sum(delta_temp,axis = 0)
                  delta.update({'delta':delta_temp})
                  #Add weights to dw DATA STRUCTURE
                  dict_temp = {'dw'+str(nh-i_l):dw,'db'+str(nh-i_l):db}
                  grad_W_ahead.update(dict_temp)
           
                #gradients for Input layer :
                delta_i  = (delta['delta'].dot(w_ahead['W2_ahead'].T)) * derivative(a['A1'],choice)
                dwi = np.dot(batch_X.T,delta_i)
                dbi = np.sum(delta_i,axis = 0)
                dict_temp = {'dw1':dwi,'db1':dbi}
                grad_W_ahead.update(dict_temp)
           
                # gradients added in grad_w_ahead
                

                #Dynamic update 
                for i in range(nh+1):
                   update_temp_w = -epsilon * grad_W_ahead['dw'+str(i+1)] /batch + momentum * prev_update['prev_update_w'+str(i+1)]
                   update_temp_b = -epsilon * grad_W_ahead['db'+str(i+1)] /batch + momentum * prev_update['prev_update_b'+str(i+1)]                                               
                   update_.update({'update_W'+str(i+1):update_temp_w})
                   update_.update({'update_b'+str(i+1):update_temp_b})

           elif opt=='adam':
               
               ###Initialise first and second moments :::
               t = t+1    
               
               
               for i in range(nh+1):
                   temp_m = (beta_1 * first_o_m['m_w'+str(i+1)] + (1-beta_1) * grad_W['dw'+str(i+1)]/batch)
                   first_o_m.update({'m_w'+str(i+1):temp_m})
                   temp_b = (beta_1 * first_o_m['m_b'+str(i+1)] + (1-beta_1) * grad_W['db'+str(i+1)]/batch)
                   first_o_m.update({'m_b'+str(i+1):temp_b})
                   
                   temp_m = (beta_2 * second_o_v['v_w'+str(i+1)] + (1-beta_2) * np.power(grad_W['dw'+str(i+1)]/batch,2))
                   second_o_v.update({'v_w'+str(i+1):temp_m})
                   temp_b = (beta_2 * second_o_v['v_b'+str(i+1)] + (1-beta_2) * np.power(grad_W['db'+str(i+1)]/batch,2))
                   second_o_v.update({'v_b'+str(i+1):temp_b})
                   
               for i in range(nh+1):
                   temp_w = -epsilon*(first_o_m['m_w'+str(i+1)] / (1-beta_1**t))/np.sqrt(second_o_v['v_w'+str(i+1)]/(1-beta_2**t) + smoothing_factor)
                   temp_b = -epsilon*(first_o_m['m_b'+str(i+1)] / (1-beta_1**t))/np.sqrt(second_o_v['v_b'+str(i+1)]/(1-beta_2**t) + smoothing_factor)
                   update_.update({'update_W'+str(i+1):temp_w,'update_b'+str(i+1):temp_b})
                   
               # calculating adam updates
               
           for i in range(nh+1):
               temp_w =  model_W['W'+str(i+1)] + update_['update_W'+str(i+1)]
               model_W.update({'W'+str(i+1):temp_w})    
               temp_b =  model_W['B'+str(i+1)] + update_['update_b'+str(i+1)]
               model_W.update({'B'+str(i+1):temp_b})
               
 
           
           for i in range(nh+1):
               prev_update.update({'prev_update_w'+str(i+1):update_['update_W'+str(i+1)],'prev_update_b'+str(i+1):update_['update_b'+str(i+1)] })

           step=step+1
           
           #After every 100th update save model parameters :
           if step % 100 == 0 :
            # Assign new parameters to the model......#pass activation and loss also
            #logging performance on training data
            
              model_data_train = { 'nh':nh,'X':X, 'y':y, 'num_datapoints':num_datapoints, 'activation':choice, 'loss': loss_, 'if_printPredictions':False} 
              for i in range(nh+1):
                  model_data_train.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })
            #return error as well as loss
              loss, accuracy = calculate_loss(model_data_train)
              print "accuracy on train :"+ str(accuracy)
              print "Loss on train :"+ str(loss)
              train_accuracy=accuracy
              
              folt.write("Epoch %i, Step %i, Loss: %f, lr: %f\n" %(itr,step, loss, epsilon))
              foat.write("Epoch %i, Step %i, Error: %i, lr: %f\n" %(itr,step, 100-accuracy, epsilon))
              
              model_data_valid = { 'nh':nh,'X':valid_set_x, 'y':valid_set_y, 'num_datapoints':len(valid_set_x), 'activation':choice, 'loss': loss_, 'if_printPredictions':False}
              for i in range(nh+1):
                  model_data_valid.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })
            #return error as well as loss
              loss, accuracy = calculate_loss(model_data_valid)
              print "accuracy on valid :"+str(accuracy)
              print "Loss on valid :"+ str(loss)
              folv.write("Epoch %i, Step %i, Loss: %f, lr: %f\n" %(itr,step, loss, epsilon))
              foav.write("Epoch %i, Step %i, Error: %i, lr: %f\n" %(itr,step, 100-accuracy, epsilon))
              
              model_data_test = { 'nh':nh,'X':test_set_x, 'y':test_set_y, 'num_datapoints':len(test_set_x), 'activation':choice, 'loss': loss_, 'if_printPredictions':False}
              for i in range(nh+1):
                  model_data_test.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })         
            #return error as well as loss
              loss, accuracy = calculate_loss(model_data_test)
              print "accuracy on test :"+str(accuracy)
              print "Loss on test :"+ str(loss)
              fols.write("Epoch %i, Step %i, Loss: %f, lr: %f\n" %(itr,step, loss, epsilon))
              foas.write("Epoch %i, Step %i, Error: %i, lr: %f\n" %(itr,step, 100-accuracy, epsilon))
              
       #Next Epoch starts here
       if if_anneal :
           model_data_valid = { 'nh':nh,'X':valid_set_x, 'y':valid_set_y, 'num_datapoints':len(valid_set_x), 'activation':choice, 'loss': loss_, 'if_printPredictions':False}
           for i in range(nh+1):
               model_data_valid.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })
           
       #return error as well as loss
           curr_valid_loss, curr_valid_accuracy = calculate_loss(model_data_valid)
           print "accuracy on valid :"+str(curr_valid_accuracy)
       
           if curr_valid_accuracy < prev_validation_accuracy :
               epsilon = epsilon/2
               if train_accuracy==0 :
                   count_Zero_training_loss_annealed= count_Zero_training_loss_annealed+1
           #half the learning rate 
           #load the pickeled model (dictionary) and assign Wi's and bi
               model_W = model_w_backup
               prev_update = prev_update_backup
               first_o_m = first_o_m_backup
               second_o_v = second_o_v_backup
               itr = itr -1
           else:
               print "Model saved!!"
               model_w_backup = model_W
               prev_update_backup = prev_update
               first_o_m_backup = first_o_m
               second_o_v_backup = second_o_v
           #save the model
               
               prev_validation_accuracy = curr_valid_accuracy
       
        ##stopping criteria
       if count_Zero_training_loss_annealed > 5:
            itr=num_epochs

    folt.close
    foat.close
    folv.close
    foav.close
    fols.close
    foas.close
    
    ## Code for final predictions on validation and test data
    model_data_valid = { 'nh':nh,'X':valid_set_x, 'y':valid_set_y, 'num_datapoints':len(valid_set_x), 'activation':choice, 'loss': loss_, 'if_printPredictions':True, 'p_path': valid_predictions_path}
    for i in range(nh+1):
                  model_data_valid.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })
    #return error as well as loss
    loss, accuracy = calculate_loss(model_data_valid)
         
    model_data_test = { 'nh':nh,'X':test_set_x, 'y':test_set_y, 'num_datapoints':len(test_set_x), 'activation':choice, 'loss': loss_, 'if_printPredictions':True, 'p_path': test_predictions_path}
    for i in range(nh+1):
                  model_data_test.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })         
    #return error as well as loss
    loss, accuracy = calculate_loss(model_data_test)
             
    ##code to save final model
    
    trained_model = {}
    for i in range(nh+1):
            trained_model.update({'W'+str(i+1):model_W['W'+str(i+1)] ,'B'+str(i+1):model_W['B'+str(i+1)] })
            
    pickle.dump( trained_model, open( args.save_dir+"save.p", "wb" ) )
    
    return model    

if __name__ == '__main__':
    args = FLAGS.parse_args()
    model = build_model(args)
    
