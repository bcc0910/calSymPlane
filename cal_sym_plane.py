from sklearn.neighbors import KDTree
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import cv2
from  obj_utils import save_obj_v
from  obj_utils import OBJLoader

class kdtree(object):
    def __init__(self):
        self.leaf_size = 2
        self.k = 1

    def build(self, X):
        self.tree = KDTree(X, leaf_size=self.leaf_size)

    def find(self, Y):
        dist, ind = self.tree.query(Y, k=self.k)
        ind=np.squeeze(np.array(ind))
        dist=np.squeeze(np.array(dist))
        return dist,ind

    def test(self):
        np.random.seed(0)
        X = np.random.random((1000, 3))
        self.build(X)
        dist,ind=self.find(X[1:20])
        print(dist)
        print(ind)

def sym_problem(vnum,lr):
    param=tf.Variable(initial_value=np.array([1,0,0,1]),trainable=True,dtype=tf.float32,name='param')
    A,B,C,D=tf.unstack(param,axis=-1)

    input=tf.placeholder(dtype=tf.float32,shape=[vnum,3],name='input')
    x,y,z=tf.unstack(input,axis=-1)

    norm=tf.maximum(tf.reduce_sum(A*A+B*B+C*C),1e-8)
    s=(A*x+B*y+C*z+D)/norm
    X = x - 2 * s * A
    Y = y - 2 * s * B
    Z = z - 2 * s * C
    output=tf.stack([X,Y,Z],axis=-1)

    corres = tf.placeholder(dtype=tf.float32, shape=[vnum, 3], name='corres')
    weight = tf.placeholder(dtype=tf.float32, shape=[vnum], name='input')

    loss_all = tf.reduce_sum(tf.abs(output-corres),axis=-1)*weight
    loss = tf.reduce_sum(loss_all)/tf.maximum(tf.reduce_sum(weight),1.0)

    tvar=tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(lr)
    opt = optimizer.minimize(loss, var_list=tvar)
    print('===='*5)
    for v in tvar:
        print(v.op.name,v.shape)
    print('===='*5)
    print(input)
    print(output)
    print(corres)
    print(weight)

    nodes={}
    nodes['opt']=opt
    nodes['loss'] = loss
    nodes['output'] = output

    nodes['input'] = input
    nodes['corres'] = corres
    nodes['weight'] = weight
    nodes['param'] = param

    return nodes

def get_context():
    tf.set_random_seed(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    return sess

def save_sym_pair():
    mobj=OBJLoader()
    mobj.load_obj("testdata/ff.obj")
    v=mobj.v

    mobj2=OBJLoader()
    mobj2.load_obj("testdata/99_out.obj")
    v2=mobj2.v

    mkd=kdtree()
    mkd.build(v)
    dist, ind = mkd.find(v2)
    f=open('testdata/sym.txt','w')
    vnum=ind.shape[0]
    f.write('%d\n' % (vnum))
    for i in range(vnum):
        f.write('%d %d\n'%(i,ind[i]))
    f.close()

def save_neis():
    mobj=OBJLoader()
    mobj.load_obj("testdata/ff.obj")
    v=mobj.v

    mkd=kdtree()
    mkd.k=5
    mkd.build(v)
    dist, ind = mkd.find(v)

    f=open('testdata/nei.txt','w')
    vnum=ind.shape[0]
    f.write('%d\n' % (vnum))
    for i in range(vnum):
        f.write('%d %d %d %d\n'%(ind[i,1],ind[i,2],ind[i,3],ind[i,4]))
    f.close()




if __name__=='__main__':
    # save_sym_pair()
    save_neis()
    assert (0)

    np.random.seed(0)
    mobj=OBJLoader()
    mobj.load_obj("testdata/ff.obj")
    v=mobj.v
    vnum=v.shape[0]

    mkd=kdtree()
    mkd.build(v)
    nodes=sym_problem(vnum=vnum,lr=0.1)
    opt=nodes['opt']
    loss=nodes['loss']
    output=nodes['output']
    inputs=nodes['input']
    corres=nodes['corres']
    weight=nodes['weight']
    param=nodes['param']

    sess=get_context()

    savedir='testdata/save/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    Nt=100
    for i in range(Nt):
        [out]=sess.run([output],feed_dict={inputs:v})
        dist,ind=mkd.find(out)
        meandist=np.mean(dist)*2.0
        w=(dist<meandist)*1.0
        nearv=v[ind,:]
        [_,ls,p] = sess.run([opt,loss,param], feed_dict={inputs: v,corres:nearv, weight:w } )       
        #print('meandist:%.1f'%(meandist))
        print(i,'validNum:%d, ls:%.2f ,p:[%.3f,%.3f,%.3f,%.3f]'%(np.sum(w),ls,p[0],p[1],p[2],p[3]))
        
        if i==0 or i==Nt-1:
            save_obj_v(out,savedir+str(i)+"_out.obj")
            save_obj_v(nearv,savedir+str(i)+"_near.obj")
        
