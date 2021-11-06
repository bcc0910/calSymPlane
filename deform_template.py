from sklearn.neighbors import KDTree
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import cv2
from  obj_utils import save_obj_v
from  obj_utils import OBJLoader
import sys

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

def rigist_problem(meanV,lr1,lr2,pairIds,NeiIds):
    param=tf.Variable(initial_value=np.array([0.0001,0.0001,0.0001,0,0,0,1,1,1]),trainable=True,dtype=tf.float32,name='param')
    ex,ey,ez,tx,ty,tz,scalex,scaley,scalez=tf.unstack(param,axis=-1)

    vnum=meanV.shape[0]
    mean_geo=tf.convert_to_tensor(meanV,dtype=tf.float32)
    offset_geo=tf.Variable(initial_value=np.zeros([vnum,3]),trainable=False,dtype=tf.float32,name='geo')

    x0,y0,z0 = tf.unstack(mean_geo, axis=-1)
    dx, dy, dz = tf.unstack(offset_geo, axis=-1)

    x = scalex * x0 + dx
    y = scaley * y0 + dy
    z = scalez * z0 + dz

    X = x * tf.cos(ez) + y * tf.sin(-ez)
    Y = x * tf.sin(ez) + y * tf.cos(ez)
    X = X * tf.cos(ey) + z * tf.sin(ey)
    Z = X * tf.sin(-ey) + z * tf.cos(ey)
    Y = Y * tf.cos(ex) + Z * tf.sin(-ex)
    Z = Y * tf.sin(ex) + Z * tf.cos(ex)
    Geo = tf.stack([X+tx,Y+ty,Z+tz], axis=-1)

    corres = tf.placeholder(dtype=tf.float32, shape=[vnum, 3], name='corres')
    weight = tf.placeholder(dtype=tf.float32, shape=[vnum], name='input')

    leftIds=pairIds[:,0]
    rightIds=pairIds[:,1]
    flipWeight= np.array([[-1,1,1]],np.float32)
    sym_loss=tf.gather(offset_geo,leftIds)*flipWeight -tf.gather(offset_geo,rightIds)
    sym_loss=tf.reduce_mean(tf.reduce_sum(tf.abs(sym_loss),axis=-1))


    nIds=np.hsplit(NeiIds,4)
    nIds= [np.squeeze(nid) for nid in nIds]
    print(nIds[0].shape)

    nei_offset=tf.gather(offset_geo, nIds[0])+tf.gather(offset_geo, nIds[1])+tf.gather(offset_geo, nIds[2])+tf.gather(offset_geo, nIds[3])
    nei_loss = tf.gather(offset_geo, leftIds) - 0.25*nei_offset
    nei_loss=tf.reduce_mean(tf.reduce_sum(tf.square(4*nei_loss),axis=-1))

    loss_all = tf.reduce_sum(tf.square(Geo-corres),axis=-1)*weight
    rec_loss = tf.reduce_sum(loss_all)/tf.maximum(tf.reduce_sum(weight),1.0)

    reg_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(offset_geo),axis=-1))

    loss=rec_loss+30*(sym_loss+0.5*nei_loss) +0.1*reg_loss

    optimizer1 = tf.train.AdamOptimizer(lr1)
    opt_rt = optimizer1.minimize(rec_loss, var_list=[param])

    optimizer2 = tf.train.AdamOptimizer(lr2)
    opt_offset = optimizer2.minimize(loss, var_list=[offset_geo])

    print(Geo)
    print(corres)
    print(weight)


    nodes={}
    nodes['opt_rt']=opt_rt
    nodes['opt_offset'] = opt_offset
    nodes['loss'] = loss
    nodes['sym_loss'] = sym_loss
    nodes['rec_loss'] = rec_loss
    nodes['nei_loss'] = nei_loss
    nodes['output'] = Geo
    nodes['reg_loss'] = reg_loss

    nodes['corres'] = corres
    nodes['weight'] = weight
    nodes['param'] = param

    return nodes

def get_context():
    tf.set_random_seed(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def train(sess,cloudfile,savedir,meanv,nodes):

    if '.obj' in cloudfile:
        mobj = OBJLoader()
        mobj.load_obj(cloudfile)
        cloud = mobj.v.copy()
    elif '.bin' in cloudfile:
        v=np.fromfile(cloudfile,dtype=np.float32)
        v=np.reshape(v,[v.shape[0]//3,3])
        mask=v[:,2]>0
        nv=v[mask,:]
        print(nv.shape)
        cloud=nv
    else:
        assert (0)

    mkd = kdtree()
    mkd.build(cloud)


    Nt=600
    vnum=meanv.shape[0]

    opt_rt=nodes['opt_rt']
    opt_offset = nodes['opt_offset']
    loss=nodes['loss']
    sym_loss=nodes['sym_loss']
    rec_loss=nodes['rec_loss']
    nei_loss=nodes['nei_loss']
    reg_loss=nodes['reg_loss']

    output=nodes['output']

    corres=nodes['corres']
    weight=nodes['weight']
    param=nodes['param']


    import time
    t1 = time.time()
    for i in range(Nt):
        [out] = sess.run([output])
        if i % 3 == 0:
            dist, ind = mkd.find(out)

        if i % 50 == 0:
            valid = np.zeros([vnum])
            inds = list(set(ind))
            id_dists = {}
            c = np.arange(ind.shape[0])
            for j in inds:
                inds = c[ind == j]
                ii = np.argsort(dist[inds])
                need_id = 1
                if i > 200:
                    need_id = 4
                eid = min(need_id, len(ii))
                valid[inds[ii[:eid]]] = 1

        meandist = np.mean(dist) * 12
        w = (dist < meandist) * 1.0
        w = valid * w

        nearv = cloud[ind, :]
        if i < 50 or i % 4 in [0, 2]:
            [_, ls, p] = sess.run([opt_rt, rec_loss, param], feed_dict={corres: nearv, weight: w})
            
            if i%30==0:
                print(i, 'validNum:%d, ls:%.2f ,p:[%.3f,%.3f,%.3f;%.3f,%.3f,%.3f;%.3f,%.3f,%.3f]' % (np.sum(w), ls, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))
        else:
            [ls] = sess.run([loss], feed_dict={corres: nearv, weight: w})
            [_, sym_ls, rec_ls, nei_ls, p, reg_ls] = sess.run(
                [opt_offset, sym_loss, rec_loss, nei_loss, param, reg_loss], feed_dict={corres: nearv, weight: w})
            if i%30==0:
                print(i,'validNum:%d, ls:%.2f [%.2f,%.2f,%.2f,%.2f] ,p:[%.3f,%.3f,%.3f ; %.3f,%.3f,%.3f; %.3f,%.3f,%.3f]' %(np.sum(w), ls, sym_ls, rec_ls, nei_ls, reg_ls, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))
    
        if i == -1 or i == Nt - 1 or i % 50 == -1:
            pass
            
        if i==Nt-1:
            w = (dist < 10) * 1.0
            w = valid * w
            [rec_ls,p] = sess.run([ rec_loss,param],feed_dict={corres: nearv, weight: w})
            if rec_ls>2.3: break
            
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            os.system('rm %s/*'%(savedir))
    
            save_obj_v(out, savedir + str(i) + "_out.obj")
            save_obj_v(nearv, savedir + str(i) + "_near.obj")

            f=open(savedir+'_detail.txt','w')
            f.write('rec_loss:%.2f\n'%(rec_ls))
            f.write('param:[%.3f,%.3f,%.3f ; %.3f,%.3f,%.3f; %.3f,%.3f,%.3f]\n' % ( p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))
            f.close()

    dt = time.time() - t1
    print('dt', int(dt * 1000))

def get_config_data():
    """
    pairIds=np.loadtxt('testdata/save_ff/sym.txt',dtype=np.int32,skiprows=1)
    NeiIds = np.loadtxt('testdata/save_ff/nei.txt', dtype=np.int32, skiprows=1)
    mobj=OBJLoader()
    mobj.load_obj("testdata/ff.obj")
    meanv=mobj.v.copy()
    nodes=rigist_problem(meanV=meanv,lr1=0.06,lr2=0.5,pairIds=pairIds,NeiIds=NeiIds)
    sess=get_context()
    """

    pairIds=np.loadtxt('config/sym.txt',dtype=np.int32,skiprows=1)
    NeiIds = np.loadtxt('config/nei.txt', dtype=np.int32, skiprows=1)
    mobj=OBJLoader()
    mobj.load_obj("config/addf.obj")

    meanv=mobj.v.copy()
    nodes=rigist_problem(meanV=meanv,lr1=0.06,lr2=0.5,pairIds=pairIds,NeiIds=NeiIds)
    sess=get_context()
    
    return sess,nodes,meanv



if __name__=='__main__':
    if len(sys.argv)==1:
        print('use: py binfile(list/txt/0.txt) gpu')
        exit(0)
    binfile=sys.argv[1] 
    devices=sys.argv[2]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
    
    np.random.seed(0)
    sess, nodes, meanv=get_config_data()
    init_op=[tf.global_variables_initializer(),tf.local_variables_initializer()]
    sess.graph.finalize()

    root_savedir="save3/"
    if len(sys.argv)>3:
        root_savedir=sys.argv[3]
        
    f=open(binfile,'r')
    lines=f.readlines()
    f.close()
    objs=[item.strip() for item in lines]

    for i,cloudfile in enumerate(objs):
        savedir = root_savedir + '/' + os.path.basename(cloudfile)+ '/'
        if '.bin' in savedir:
            savedir=savedir.replace('.bin','')
        if '.obj' in savedir:
            savedir=savedir.replace('.obj','')
        sess.run(init_op)
        train(sess, cloudfile, savedir, meanv, nodes)
        print(i)
        print('========='*10)
