import tensorflow as tf
from base_function import *
from visualization import *
import vgg16
import random
# import for dqn
from collections import deque
import os
eta=3
tau=0.65
trigger=False #need to init for every episode
number_steps=10
number_actions=5
actions_index=range(5)
actions=['lu','ld','ru','rd','c']
percent=[0.9,0.9]
# global variables for dqn
N_exp_replay=1000
gamma=0.90
init_epsilon=1.0
final_epsilon=0.1
epsilon_reduce=50
C_target_update=1000
batch_size=100
N_start_train=1000 # experiences replay
N_save_net=100
total_episodes=60
#images_path="test_bird"
images_path="./test_data/"
# weight-bias initialize defination
def weight_varibale(shape):
    initial=tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.01,shape=shape)
    return tf.Variable(initial)
# net aritechture create
def create_q_network():
    w_fc1,b_fc1=weight_varibale([25112,1024]),bias_variable([1024])
    w_fc2,b_fc2=weight_varibale([1024,1024]),bias_variable([1024])
    w_fc3,b_fc3=weight_varibale([1024,number_actions]),bias_variable([number_actions])
    input=tf.placeholder("float32",[None,25112])   #  dim?????
    h_fc1 = tf.nn.relu(tf.matmul(input, w_fc1) + b_fc1)
    h_fc1=tf.nn.dropout(h_fc1,0.5)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
    h_fc1 = tf.nn.dropout(h_fc2,0.5)
    prob_action=tf.matmul(h_fc1, w_fc3) + b_fc3
    return input,prob_action
# net train
def train_q_network(input,prob_action,sess):
    sess1 = tf.Session()
    vgg = vgg16.Vgg16()
    #-----------net definition-----------
    yj_form_arg=tf.placeholder("float",[None])
    aj_form_arg=tf.placeholder("float",[None,number_actions])
    Q_value=tf.reduce_sum(tf.multiply(prob_action,aj_form_arg),reduction_indices=1)
    loss=tf.reduce_mean(tf.square(yj_form_arg-Q_value))
    train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
    sess.run(tf.global_variables_initializer())
    #-------------load to test-----------
    checkpoint=tf.train.get_checkpoint_state("savenetworks")
    if checkpoint and checkpoint.model_checkpoint_path:
        tf.train.Saver().restore(sess,checkpoint.model_checkpoint_path)
        train_index=0
        print("forward evaluate: successfully loaded:",checkpoint.model_checkpoint_path)
    else:
        print("backward train: could not find old network weights")
        train_index=1
    D=deque()
    learning_path=[]
    C_index=0
    pro_j1_action=np.zeros([batch_size,number_actions])
    epsilon=init_epsilon
    t=0
    for image_name in os.listdir(images_path):
        image_path='test_data/'+image_name
        print('episode for ', image_path, ' image-env.', 'epsilon: ')
        img,image_shape = load_image(image_path,init=False)
        b=[0,0,224,224]
        g=[96,77,260,170]
        learning_path.append(b)
        g=transform_ordi(g,image_shape[0],image_shape[1],'o_t_s')
        initialize(img,g)
        st_image=extract_feature(img,sess1,vgg)
        st_image,history=st_image.eval(session=sess),np.zeros([24,1])
        st=np.vstack((st_image,history))
        trigger = False
        counter=0
        while (trigger==False):
            if random.random()<epsilon:
                action_index=random.randrange(number_actions)
                print('random action',action_index)
            else:
                action_index=np.argmax(prob_action.eval(feed_dict={input: [st]}))
                print('calu action',action_index)
            action=actions[action_index]
            print(action)
            action_reward=0
            b1=get_rect_after_action(b,b,action,percent)
            action_reward=reward_calu(b,b1,g,tau,eta,trigger)
            print('----b1----',b1)
            #st1_img = crop_image(img, b1)
            #st1_img=st1_img.resize((224,224))
            #st1_image=extract_feature(st1_img,sess1,vgg)
            #history=get_history(history,action_index)
            #st1=np.vstack((st1_image,history))
            learning_path.append(b1)

            b=b1
            counter+=1
            if counter>=number_steps:
                trigger=True
            '''
            D.append((st,action_index,action_reward,st1))
            if len(D)>N_exp_replay: D.popleft()
            st=st1
            C_index+=1
            if train_index==1:
                 if len(D)>N_start_train:
                    minibatch=random.sample(D,batch_size)
                    st=[d[0] for d in minibatch]
                    aj=[d[1] for d in minibatch]
                    rj=[d[2] for d in minibatch]
                    st1=[d[3] for d in minibatch]
                    if C_index >= C_target_update:
                       pro_j1_action = prob_action.eval(feed_dict={input:st1})
                       C_index = 0
                    yj=[]
                    for i in range(0,len(minibatch)):
                       terminal=minibatch[i][4]
                       if not terminal: yj.append(rj[i])
                       else: yj.append(rj[i]+gamma*np.max(pro_j1_action[i]))
                    train_step.run(feed_dict={
                       input:st,
                       aj_form_arg:aj,
                       yj_form_arg:yj})
                 t=t+1
                 if t%epsilon_reduce==0 and epsilon>= final_epsilon:
                    epsilon-=(init_epsilon-final_epsilon)/total_episodes
                 if t%N_save_net==0:
                    tf.train.Saver().save(sess,'savenetworks/'+'dqn',global_step=t)'''
        plot_path(learning_path, img)

def main():
    sess=tf.InteractiveSession()
    input,pro_action=create_q_network()
    train_q_network(input,pro_action,sess)
main()
