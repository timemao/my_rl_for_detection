import tensorflow as tf
from PIL import Image
import math
import numpy as np
def load_image(path,init=True):
    img = Image.open(path)
    if init:
        resized_img=img
    else: resized_img = img.resize((224, 224))
    resized_img=np.array(resized_img)/255.0
    return resized_img,img.size

def extract_feature(image,sess,vgg):
    batch= image.reshape((1, 224, 224, 3))
    image = tf.placeholder("float", [None, 224, 224, 3])
    vgg.build(image)
    feature_map = sess.run(vgg.pool5, feed_dict={image: batch})
    feature_conv = tf.reshape(feature_map,(25088,1))
    return feature_conv
#reward
def reward_calu(b,b1,g,tau,eta,trigger=False):
    if trigger:
        if Iou(b,g)>tau:
            reward=eta
        else: reward=-eta
    else:
        reward=np.sign(Iou(b1,g)-Iou(b,g))
    return reward
def area_and_calu(b,g):
    if b[0]<=g[0]<=b[2] and b[1]<=g[1]<=b[3]:
        #s1
        if b[2]<=g[2] and b[3]<=g[3]: area_and=math.fabs(g[0]-b[2])*math.fabs(g[1]-b[3])
        else:
            if b[2]>=g[2] and b[3]>=g[3]: area_and=area_calu(g)
            else:
                if b[2]>g[2]: area_and = math.fabs(g[0] - g[2]) * math.fabs(g[1] - b[3])
                else: area_and = math.fabs(g[0] - b[2]) * math.fabs(g[1] - g[3])
    else:
        #s2
        if g[0]<=b[0]<=g[2] and g[1]<=b[1]<=g[3]: area_and=area_and_calu(g,b)
        else:
            #s3
            if g[0]<=b[0]<=g[2] and g[1]<=b[3]<=g[3]:
                if b[2]<=g[2] and b[1]<=g[1]: area_and=area_calu(b)
                else:
                    if b[2]>g[2] and b[1]<g[1]: area_and = math.fabs(b[0] - g[2]) * math.fabs(b[3] - g[1])
                    else:
                       if b[2]<g[2]: area_and = math.fabs(b[0] - b[2]) * math.fabs(b[3] - g[1])
                       else: area_and = math.fabs(b[0] - g[2]) * math.fabs(b[3] - b[1])
            else:
                #s4
                if b[0]<=g[0]<=b[2] and b[1]<=g[3]<=b[3]: area_and=area_and_calu(g,b)
                #no intersection
                else: area_and=0
    return area_and
def area_calu(b):
    return math.fabs(b[0]-b[2])*math.fabs(b[1]-b[3])
def Iou(b,g):
    Iou=area_and_calu(b,g)/(area_calu(b)+area_calu(g)-area_and_calu(b,g))
    return Iou
#action eff.
def get_rect_after_action(b,b_save,action,percent):
    p_e,p_c=percent[0],percent[1]
    b = np.array(b)
    b_width,b_height=b[2]-b[0],b[3]-b[0]
    result=b
    r_width, r_height = p_e * b_width, p_e * b_height
    if action=='lu':
        result[2],result[3]=b[0]+r_width,b[1]+r_height
    elif action=='ld': result[1],result[2]=b[3]-r_height,b[0]+r_width
    elif action=='ru': result[0],result[3]=b[2]-r_width,b[1]+r_height
    elif action=='rd': result[0],result[1]=b[2]-r_width,b[3]-r_height
    elif action=='c':
        result[2], result[3] = p_c* b[2], p_c* b[3]
        move_dist=get_rect_center(b_save)-get_rect_center(result)
        result[0],result[2]=result[0]+move_dist[0],result[2]+move_dist[0]
        result[1], result[3] = result[1] + move_dist[1], result[3] + move_dist[1]
    return result
def get_rect_center(b):
    center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
    center=np.array(center)
    return center
def crop_image(image,box):
    print(image)
    print(image.shape)
    image_resize=np.resize(image,(224,224))
    img=image_resize.crop(box)
    return img
def get_history(history,action_index):
    h_new=history[6:23]
    a_new=np.zeros([6,1])
    a_new[action_index]=1
    new_history=np.vstack((h_new,a_new))
    print(new_history)
    return new_history
# for test
def test_fun():
    b=[0,0,224,224]
    g=[30,30,80,80]
    print(Iou(b,g))
    print(Iou(g,b))
    print('test2')
    b2=[0,0,2,2]
    g2=[1,1,3,3]
    print(Iou(b,g))
    print(Iou(g,b))
    print('test3')
    b3=[1,0,3,2]
    g3=[0,1,2,3]
    print(Iou(b,g))
    print(Iou(g,b))
    print('reward=test:',reward_calu(b3,b3,g2,0.5,3))
    b=[0,0,1,1]
    g=[1,1,2,2]
    print(Iou(b,g))
    print(Iou(g,b))
#test_fun()
def test_fun2():
    b = [0, 0, 224, 224]
    percent=[0.5,0.7]
    print('test1:',get_rect_after_action(b,b,'ld',percent))
    print(b)
    print('test2:', get_rect_after_action(b, b, 'lu', percent))
    print(b)
    print('test3:', get_rect_after_action(b, b, 'ru', percent))
    print(b)
    print('test4:', get_rect_after_action(b, b, 'rd', percent))
    print(b)
    print('test5:', get_rect_after_action(b, b, 'c', percent))
    print(b)
    print('test6:', get_rect_after_action(b, b, 't', percent))
#test for read images
def test_fun3():
    box_path="bounding.box"
def test_fun4():
    history=np.zeros([24,1])
    action_index=3
    get_history(history,action_index)