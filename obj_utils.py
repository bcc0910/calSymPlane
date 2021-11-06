import numpy as np

def save_obj_v_vn_f(v,vn,f,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f\n' %(v[i,0],v[i,1],v[i,2]))
        for i in range(vn.shape[0]):
            fid.write('vn %f %f %f\n' %(vn[i,0],vn[i,1],vn[i,2]))
        for i in range(f.shape[0]):
            fid.write('f %d//%d %d//%d %d//%d\n' %(f[i,0]+1,f[i,0]+1,f[i,1]+1,f[i,1]+1,f[i,2]+1,f[i,2]+1))
    print('save %s success' %(name))


def save_obj_v_vc(v,vc,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f %f %f %f\n' %(v[i,0],v[i,1],v[i,2],vc[i,0],vc[i,1],vc[i,2]))
    print('save %s success' %(name))

def save_obj_v_vc_vn(v,vc,vn,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            if np.sum(v[i,:])!=0.0:
                fid.write('v %f %f %f %d %d %d\n' %(v[i,0],v[i,1],v[i,2],vc[i,0],vc[i,1],vc[i,2]))
                fid.write('vn %f %f %f\n' %(vn[i,0],vn[i,1],vn[i,2]))
    print('save %s success' %(name))


def save_obj_v_f(v,f,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f\n' %(v[i,0],v[i,1],v[i,2]))
        for i in range(f.shape[0]):
            fid.write('f %d %d %d\n' %(f[i,0]+1,f[i,1]+1,f[i,2]+1))
    print('save %s success' %(name))

def save_obj_v_f_vc(v,vc,f,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f %f %f %f\n' %(v[i,0],v[i,1],v[i,2],vc[i,0],vc[i,1],vc[i,2]))
        for i in range(f.shape[0]):
            fid.write('f %d %d %d\n' %(f[i,0]+1,f[i,1]+1,f[i,2]+1))
    print('save %s success' %(name))

def save_obj_v(v,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f 0 0 1\n' %(v[i,0],v[i,1],v[i,2]))
    #print('save %s success' %(name))


def save_valid_vfn(v, vn, f, name, valid):
    with open(name, 'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(vn.shape[0]):
            fid.write('vn %f %f %f\n' % (vn[i, 0], vn[i, 1], vn[i, 2]))
        for i in range(f.shape[0]):
            ind1 = f[i, 0]
            ind2 = f[i, 1]
            ind3 = f[i, 2]
            if valid[ind1] > 0 and valid[ind2] > 0 and valid[ind3] > 0:
                fid.write('f %d//%d %d//%d %d//%d\n' % (ind1 + 1, ind1 + 1, ind2 + 1, ind2 + 1, ind3 + 1, ind3 + 1))
    print('save %s success' % (name))


def save_vfnc(v, f, vn, vc, name):
    with open(name, 'w') as fid:
        for i in range(v.shape[0]):
            if np.sum(vc[i, :])==0:
                vc[i, :] =150
            Ind=vc[i,:]>255
            vc[i,Ind]=255
            fid.write('v %f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], vc[i, 0], vc[i, 1], vc[i, 2]))
        for i in range(vn.shape[0]):
            fid.write('vn %f %f %f\n' % (vn[i, 0], vn[i, 1], vn[i, 2]))
        for i in range(f.shape[0]):
            ind1 = f[i, 0]
            ind2 = f[i, 1]
            ind3 = f[i, 2]
            fid.write('f %d//%d %d//%d %d//%d\n' % (ind1 + 1, ind1 + 1, ind2 + 1, ind2 + 1, ind3 + 1, ind3 + 1))
    print('save %s success' % (name))

def save_valid_vfnc(v, f, vn, vc, name, valid):
    with open(name, 'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], vc[i, 0], vc[i, 1], vc[i, 2]))
        for i in range(vn.shape[0]):
            fid.write('vn %f %f %f\n' % (vn[i, 0], vn[i, 1], vn[i, 2]))
        for i in range(f.shape[0]):
            ind1 = f[i, 0]
            ind2 = f[i, 1]
            ind3 = f[i, 2]
            if valid[ind1] > 0 and valid[ind2] > 0 and valid[ind3] > 0:
                fid.write('f %d//%d %d//%d %d//%d\n' % (ind1 + 1, ind1 + 1, ind2 + 1, ind2 + 1, ind3 + 1, ind3 + 1))
    print('save %s success' % (name))

def gen_cloudFace(w, h):
    col = w - 2
    row = h - 2
    x = np.array(range(0, col), dtype=np.int32)
    y = np.array(range(0, row), dtype=np.int32)
    xx = np.tile(np.expand_dims(x, axis=0), (row, 1))
    yy = np.tile(np.expand_dims(y, axis=-1), (1, col))

    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])

    ind0 = yy * w + xx
    ind1 = (yy + 1) * w + xx
    ind2 = (yy + 1) * w + (xx + 1)
    ind3 = yy * w + (xx + 1)
    f1 = np.concatenate([ind0, ind1, ind3], axis=-1)
    f2 = np.concatenate([ind1, ind2, ind3], axis=-1)
    cloudface = np.concatenate([f1, f2], axis=0)

    print("cloudface", cloudface.shape)
    return cloudface

class OBJLoader(object):
    def __init__(self):
        self.need_v=True
        self.need_f=True
        self.need_vn= False
        self.need_vt = False

    def load_obj(self,name):
        tags=[]
        if self.need_v:
            tags.append('v')
        if self.need_f:
            tags.append('f')
        if self.need_vn:
            tags.append('vn')
        if self.need_vt:
            tags.append('vt')

        fid=open(name, 'r')
        lines = fid.readlines()
        fid.close()

        if len(lines)==0:
            print('%s not exist!'%(name))
            return

        vs = []
        fs = []
        vns = []
        vts = []
        for line in lines:
            tag = line.split(' ')[0]
            if tag not in tags:
                continue

            line = line.strip()
            if tag == 'v':
                x = line.split(' ')[1]
                y = line.split(' ')[2]
                z = line.split(' ')[3]

                vs.append(float(x))
                vs.append(float(y))
                vs.append(float(z))
            elif tag == 'f':
                if len(line.split(' '))==4:
                    x = line.split(' ')[1]
                    y = line.split(' ')[2]
                    z = line.split(' ')[3]
                    if '/' not in line:
                        fs.append(int(x)-1)
                        fs.append(int(y)-1)
                        fs.append(int(z)-1)
                    else:
                        fs.append(int(x.split('/')[0]) - 1)
                        fs.append(int(y.split('/')[0]) - 1)
                        fs.append(int(z.split('/')[0]) - 1)

                elif len(line.split(' '))==5:
                    x = line.split(' ')[1]
                    y = line.split(' ')[2]
                    z = line.split(' ')[3]
                    z2= line.split(' ')[4]

                    if '/' not in line:
                        fs.append(int(x) - 1)
                        fs.append(int(y) - 1)
                        fs.append(int(z) - 1)

                        fs.append(int(x) - 1)
                        fs.append(int(z) - 1)
                        fs.append(int(z2) - 1)
                    else:
                        fs.append(int(x.split('/')[0]) - 1)
                        fs.append(int(y.split('/')[0]) - 1)
                        fs.append(int(z.split('/')[0]) - 1)

                        fs.append(int(x.split('/')[0]) - 1)
                        fs.append(int(z.split('/')[0]) - 1)
                        fs.append(int(z2.split('/')[0]) - 1)
                else:
                    print(len(line.split(' ')),line.split(' '))
                    assert (0)

            elif tag == 'vn':
                x = line.split(' ')[1]
                y = line.split(' ')[2]
                z = line.split(' ')[3]

                vns.append(float(x))
                vns.append(float(y))
                vns.append(float(z))
            elif tag == 'vt':
                x = line.split(' ')[1]
                y = line.split(' ')[2]

                vts.append(float(x))
                vts.append(float(y))

        print("#v:%d , #f:%d " % (len(vs) / 3, len(fs) / 3))
        self.v = np.asarray(vs, dtype=np.float32).reshape(-1, 3)
        self.f = np.asarray(fs, dtype=np.int32).reshape(-1, 3)
        self.vn = np.asarray(vns, dtype=np.float32).reshape(-1, 3)
        self.vt = np.asarray(vts, dtype=np.float32).reshape(-1, 2)