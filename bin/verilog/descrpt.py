

from env2 import tf, np, plt
from env2 import global_tf_float_precision
from env2 import op_module
from network import set_config, get_config
from network import read_cf, read_net, read_atoms, read_atoms_past
from network import get_dic, get_sess, get_var, read_data
from network import plot_data
import code_vcs


# Build
# =====================================================================


def build_seg(x, xps, ks, bs):
    x1 = tf.clip_by_value(x, xps[0], xps[1])
    x2 = tf.clip_by_value(x, xps[1], xps[2])
    x3 = tf.clip_by_value(x, xps[2], xps[3])

    y = (x1-xps[0])*ks[0] + (x2-xps[1])*ks[1] + (x3-xps[2])*ks[2]
    return y

def build_u2r(u):
    cf = read_cf()
    xps, ks, bs, xps2, ks2, bs2 = cf['map_param']
    xps2 = xps2[:,0].tolist()
    xps2.insert(0,0)
    xps2.append(1.0)

    r2 = build_seg(u, xps2, ks2, bs2) * (cf['ox127112']**2)

    return r2

def build_r2s(r2):
    cf = read_cf()
    limit = cf['ox127112108121118122']
    rmin = cf['ox127112128116101']
    rmax = cf['ox127112']
    ntype = cf['ox123129134125114112118123114']
    avg = cf['avg']
    std = cf['std']

    r = tf.sqrt(r2)
    r_ = tf.clip_by_value(r, rmin, rmax)
    r__ = tf.clip_by_value(r, limit, rmax) # 小于此limit的值保持恒定
    uu = (r_ - rmin) / (rmax - rmin)
    vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1

    sl = []
    srl = []

    for tt in range(ntype):
        s = vv / r__
        sr = s / r__
        s = tf.reshape(s, [-1, 1])
        sr = tf.reshape(sr, [-1, 1])
        s = (s - avg[tt,0]) / std[tt,0]
        sr = sr / std[tt,1]
        sl.append(s)
        srl.append(sr)
    return sl, srl

def build_ds_dr(r2, s, sr):
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']

    ds_drl = []
    dsr_drl = []
    for tt in range(ntype):
        si = s[tt]
        sri = sr[tt]
        ds_dr = tf.gradients(si, r2)
        dsr_dr = tf.gradients(sri, r2)
        ds_drl.append(ds_dr[0])
        dsr_drl.append(dsr_dr[0])
    return ds_drl, dsr_drl

def build_s2G(s):
    cf = read_cf()
    ntypex = cf['ox123129134125114133']
    ntype = cf['ox123129134125114112118123114']

    filter_precision = global_tf_float_precision
    activation_fn = tf.tanh
    outputs_size = cf['ox091091092081082108083082078096']
    outputs_size.insert(0, 1)

    xyz_scatters = []
    for tt in range(ntypex):
        for tt2 in range(ntype):
            xyz_scatter = s[tt]
            for ii in range(1, len(outputs_size)):
                lname = "fea_t%d_t%d_l%d_"%(tt, tt2, ii-1)
                w, b = get_var([lname+'m', lname+'b'],
                            [outputs_size[ii - 1], outputs_size[ii]],
                            filter_precision,
                            net = cf)
                if outputs_size[ii] == outputs_size[ii-1]:
                    xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
                elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                    xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
                else:
                    xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
            xyz_scatters.append(xyz_scatter)
    return xyz_scatters

def build_dG_ds(G, s):
    cf = read_cf()
    ntypex = cf['ox123129134125114133']
    ntype = cf['ox123129134125114112118123114']

    dG_ds = []
    for tt in range(ntypex):
        for tt2 in range(ntype):
            Gi = G[tt*ntype+tt2]
            si = s[tt]

            dG_ds_i = []
            for ii in range(cf['ox090062']):
                dG_ds_ii = tf.reshape(tf.gradients(Gi[:,ii], si), [-1, 1])
                dG_ds_i.append(dG_ds_ii)
            dG_ds_i = tf.concat(dG_ds_i, axis=1)
            dG_ds.append(dG_ds_i)
    return dG_ds

# Test
# =====================================================================

def test_u2G(only_build=False):
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    ntypex = cf['ox123129134125114133']
    avg = cf['avg']
    std = cf['std']
    # 分为两部分
    dic_ph = {}
    dic_ph['u'] = tf.placeholder(global_tf_float_precision, [None, 1], 't_u')
    dic_ph['r2'] = build_u2r(dic_ph['u'])
    dic_ph['s'], dic_ph['sr'] = build_r2s(dic_ph['r2'])
    dic_ph['ds_dr2'], dic_ph['dsr_dr2'] = build_ds_dr(dic_ph['r2'], dic_ph['s'], dic_ph['sr'])
    print(dic_ph['s'])
    print(dic_ph['sr'])
    print(dic_ph['ds_dr2'])
    print(dic_ph['dsr_dr2'])

    if only_build: return dic_ph
    else: sess = get_sess()

    N = 2 ** cf['ox091079086097108083082078108101']
    u = 1.0 * np.arange(0,N) / N
    u = np.reshape(u, [-1,1])
    feed_dic = {dic_ph['u']:u}
    key = 'u,r2,s,sr,ds_dr2,dsr_dr2'
    res = sess.run(get_dic(dic_ph, key), feed_dic)

    res2 = {}
    key = key.split(',')
    for ii in range(len(key)):
        res2[key[ii]] = res[ii]

    # change value
    # 将0处值设为0
    for tt in range(ntypex):
        res2['s'][tt][0] = -avg[tt,0] / std[tt,0]
        res2['sr'][tt][0] = 0
        res2['ds_dr2'][tt][0] = 0
        res2['dsr_dr2'][tt][0] = 0
    
    
    r = np.sqrt(res2['r2'])
    sess.close()

    test_u2G_2(dat=res2)

def test_u2G_2(only_build=False, dat=None):
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    ntypex = cf['ox123129134125114133']

    dic_ph = {}
    dic_ph['s'] = [tf.placeholder(global_tf_float_precision, [None, 1], 't_s%d'%tt) for tt in range(ntype)]
    dic_ph['G'] = build_s2G(dic_ph['s'])
    dic_ph['dG_ds'] = build_dG_ds(dic_ph['G'], dic_ph['s'])

    if only_build: return dic_ph
    else: sess = get_sess()

    if dat == None:
        dat = np.load('./tmp_u2G.npy', allow_pickle=True)[0]
    u = dat['u']
    r = np.sqrt(dat['r2'])
    s = dat['s']
    sr = dat['sr']
    ds_dr2 = dat['ds_dr2']
    dsr_dr2 = dat['dsr_dr2']
    feed_dic = dict(zip(dic_ph['s'], s))
    res = sess.run(get_dic(dic_ph, 'G,dG_ds'), feed_dic)

    G = res[0]
    dG_ds = res[1]

    dG_dr2 = []
    for tt in range(ntypex):
        for tt2 in range(ntype):
            dG_dr2.append(dG_ds[tt*ntype+tt2] * ds_dr2[tt])

    for tt in range(ntypex):
        s[tt][0] = np.round(s[tt][0] * 1024) / 1024
        for tt2 in range(ntype):
            G[tt*ntype+tt2][0] = np.round(G[tt*ntype+tt2][0] * 1024) / 1024

    def qqq(dat, nbit, is_round=False):
        dat = dat if type(dat) == list else [dat]
        prec = 2 ** (nbit-1)

        #
        if is_round:
            dat = [np.round(dati * prec) / prec for dati in dat]
        else:
            dat = [np.floor(dati * prec) / prec for dati in dat]
        return dat

    maps = {}
    val = qqq(s      , cf['ox091079086097108083082078108083089080068075086097072071067097072']); maps['val_s'      ] = val
    val = qqq(sr     , cf['ox091079086097108083082078108083089080068075086097072071067097072']); maps['val_sr'     ] = val
    val = qqq(ds_dr2 , cf['ox091079086097108083082078108083089080068075086097072071067097072']); maps['val_ds_dr2' ] = val
    val = qqq(dsr_dr2, cf['ox091079086097108083082078108083089080068075086097072071067097072']); maps['val_dsr_dr2'] = val
    val = qqq(G      , cf['ox091079086097108083082078108083089080068075086097072071067097072']); maps['val_G'      ] = val
    val = qqq(dG_dr2 , cf['ox091079086097108083082078108083089080068075086097072071067097072']); maps['val_dG_dr2' ] = val

    n = len(u)
    for tt in range(ntypex):
        for tt2 in range(ntype):
            postfix = '_t%d_t%d'%(tt,tt2)
            maps['table_s'      +postfix] = maps['val_s'      ][tt].reshape([n, -1])
            maps['table_sr'     +postfix] = maps['val_sr'     ][tt].reshape([n, -1])
            maps['table_ds_dr2' +postfix] = maps['val_ds_dr2' ][tt].reshape([n, -1])
            maps['table_dsr_dr2'+postfix] = maps['val_dsr_dr2'][tt].reshape([n, -1])
            maps['table_G'      +postfix] = maps['val_G'      ][tt*ntype+tt2].reshape([n, -1])
            maps['table_dG_dr2' +postfix] = maps['val_dG_dr2' ][tt*ntype+tt2].reshape([n, -1])
    sfea, sgra = code_vcs.save_fea_map(maps)
    maps['sfea'] = sfea
    maps['sgra'] = sgra
    np.save(get_config('map_file'),[maps], allow_pickle=True)

    sess.close()


# Main
# =====================================================================


def main(argvs):
    if argvs[0] == 'map':
        config_file = argvs[1]
        map_file = argvs[2]
        set_config('config_file', config_file)
        set_config('map_file', map_file)
        test_u2G()



