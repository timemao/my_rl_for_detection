from matplotlib import pyplot as plt
from base_function import *
def initialize(img,g):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    pointx,pointy=get_box(g,init=True)
    ax.plot(pointx, pointy, 'r',linewidth=2)
    plt.axis('off')
    plt.imshow(img)
    plt.pause(1)
    plt.close()
def plot_path(learning_path,img):
    k=1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for box in learning_path:
        k=k+1
        pointx, pointy = get_box(box)
        ax.plot(pointx, pointy, 'r', linewidth=math.log(k)+0.6)
        plt.axis('off')
    plt.show()
def get_box(b,init=False):
    if init:
        b[2]=b[0]+b[2]
        b[3]=b[1]+b[3]
    pointx=[b[0],b[2],b[2],b[0],b[0]]
    pointy=[b[1],b[1],b[3],b[3],b[1]]
    return pointx,pointy
def transform_ordi(b,width,height,direct):
    if direct=='s_t_o':rate_w,rate_h=width/224,height/224
    else: rate_w,rate_h=224/width,224/height
    b[0],b[2]=rate_w*b[0],rate_w*b[2]
    b[1],b[3]=rate_h*b[1],rate_h*b[3]
    return b
#test
def test_fun():
    image_path="bird.jpg"
    initialize(image_path)
    b=[0,0,224,224]
    b_save=b
    test=get_rect_after_action(b,b_save,'ld',[0.5,0.7])
    update(test,image_path)