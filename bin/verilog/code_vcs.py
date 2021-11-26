

import os
from env2 import np
from network import read_cf
import code_hb

import fio
from env2 import intt


# config
# =====================================================================

config = {
    'test_range_count':0,
    'resort_file':"idx_spe.txt",
    'idx':[]
}

def set_config(key, value):
    config[key] = value

def get_config(key):
    if key in config.keys():
        return config[key]
    else:
        return None


# config
# =====================================================================

def resort(a):
    idx = get_config('idx')
    if len(idx) == 0:
        resort_file = get_config('resort_file')
        if os.path.exists(resort_file):
            idx = np.int32(np.loadtxt(resort_file))
            set_config('idx', idx)
        else:
            return a
    else:
        N = len(idx)
        a = a.reshape([N, -1])
        a = a[idx]
    return a

# test_range
# =====================================================================

def test_range(v, name, nc=0):
    """:Test the range of v.
    v has name.
    if nc is greater than 0, reshape v and
     test per column of v.
    """
    if get_config('test_range_count') == 0:
        print("#INFO: TEST_RANGE: %20s %16s %16s %16s %5s"%("name","vmin","vmax","vmax_abs","log2_n"))
        set_config('test_range_count', 1)

    if nc > 0:
        v = np.reshape(v, [-1, nc])

        for ii in range(nc):
            vmin = np.min(v[:,ii])
            vmax = np.max(v[:,ii])
            vmax_abs = np.max(np.abs(v[:,ii]))
            n = intt(np.ceil(np.log2(vmax_abs+1)))
            print("INFO: TEST_RANGE[%d]: %20s %16.3f %16.3f %16.3f %5d"%(ii, name, vmin, vmax, vmax_abs, n))
    else:
        vmin = np.min(v)
        vmax = np.max(v)
        vmax_abs = np.max(np.abs(v))
        n = intt(np.ceil(np.log2(vmax_abs+1)))
        print("INFO: TEST_RANGE: %20s %16.3f %16.3f %16.3f %5d"%(name, vmin, vmax, vmax_abs, n))

# save_fea_map
# =====================================================================

def save_fea_map(maps):
    """
    保存feaNet的查表关系
    """
    cf = read_cf()
    ntype = cf['ox123129134125114112118123114']
    same_net = cf['ox128110122114108123114129']
    prec = 2 ** cf['ox091079086097108083082078108083089080068075086097072071067097072']

    # 顺序重排
    #**为了方便增加fea
    if same_net:
        idxs = [[0,tt] for tt in range(ntype)]
    else:
        idxs = []
        for tt in range(ntype):
            for tt2 in range(tt+1): idxs.append([tt, tt2])
            for tt2 in range(tt): idxs.append([tt-tt2-1, tt])
    
    # get value
    s       = []
    sr      = []
    ds_dr2  = []
    dsr_dr2 = []
    G       = []
    dG_dr2  = []

    for ii in range(len(idxs)):
        tt, tt2 = idxs[ii]
        print(ii, tt, tt2)
        postfix = '_t%d_t%d'%(tt,tt2)
        s       .append( maps['table_s'      +postfix] )
        sr      .append( maps['table_sr'     +postfix] )
        ds_dr2  .append( maps['table_ds_dr2' +postfix] )
        dsr_dr2 .append( maps['table_dsr_dr2'+postfix] )
        G       .append( maps['table_G'      +postfix] )
        dG_dr2  .append( maps['table_dG_dr2' +postfix] )

    # stack the value of different type of neighbor atoms
    s       = np.vstack(s       )
    sr      = np.vstack(sr      )
    ds_dr2  = np.vstack(ds_dr2  )
    dsr_dr2 = np.vstack(dsr_dr2 )
    G       = np.vstack(G       )
    dG_dr2  = np.vstack(dG_dr2  )

    # quantify
    s       = np.round(s       * prec)
    sr      = np.round(sr      * prec)
    ds_dr2  = np.round(ds_dr2  * prec)
    dsr_dr2 = np.round(dsr_dr2 * prec)
    G       = np.round(G       * prec)
    dG_dr2  = np.round(dG_dr2  * prec)


    # fea
    dat = [s, sr, G]
    dat = np.hstack(dat)
    dat = dat[:,::-1]
    hs = code_hb.arr2hex(dat, cf['ox091079086097108083082078'], nmerge=(cf['ox090062']+2)*cf['ox091079086097108083082078'],signed=True)
    sfea = hs

    # grad
    dat = [ds_dr2, dsr_dr2, dG_dr2]
    dat = np.hstack(dat)
    dat = dat[:,::-1]
    hs = code_hb.arr2hex(dat, cf['ox091079086097108083082078'], nmerge=(cf['ox090062']+2)*cf['ox091079086097108083082078'],signed=True)
    sgra = hs

    #
    return sfea, sgra


# save_descrpt
# =====================================================================

def save_descrpt(res):
    """
    res: o_rmat,o_rmat_deriv,o_rij,o_nlist
    """
    cf = read_cf()
    descrpt = res['o_rmat']
    descrpt_deriv = res['o_rmat_deriv']
    nlist = intt(res['o_nlist']).reshape([cf['ox091110'], cf['ox091086']])
    rij = res['o_rij']
    spe = np.reshape(res['t_type'], [-1, 1])
    crd = np.reshape(res['t_coord'], [-1, 3])

    save_R4(descrpt)
    save_lst(spe, nlist)
    save_rij(rij)

def save_R4(R4):
    cf = read_cf()
    Na = cf['ox091110']
    prec1 = 2**cf['ox091079086097108081078097078108083089']
    R4 = np.round(np.reshape(R4, [Na, -1]) * prec1)
    test_range(R4, 'R4')

def save_lst(spe, lst):
    cf = read_cf()
    Na = cf['ox091110']
    NI = cf['ox091086']
    na, ni = lst.shape
    if ni < NI:
        lst = np.hstack([lst, np.zeros([Na, NI-ni])-1])

    for ii in range(Na):
        for jj in range(NI):
            if lst[ii,jj] == -1:
                lst[ii,jj] = ii
    np.savetxt('./out/lst.txt', lst.reshape([Na, -1]), fmt='%4d')

    spe2 = spe.reshape(-1)
    spe2 = spe2[lst]
    np.savetxt('./out/spe.txt', spe2, fmt='%4d')

def save_rij(rij):
    cf = read_cf()
    Na = cf['ox091110']
    rij_ = np.round(rij * (2**cf['ox091079086097108081078097078108083089']))
    rij_ = np.reshape(rij_, [Na, -1])
    rij = np.reshape(rij, [Na, -1])
    np.savetxt('./out/rij.txt', rij_, fmt="%8d")
    np.savetxt('./out/rij_float.txt', rij, fmt="%8.5f")



# save_cal_fij
# =====================================================================

def save_cal_fij(res):
    """
    """
    print("INFO: Save CAL_FIJ")
    cf = read_cf()
    Na = cf['ox091110']
    NI = cf['ox091086']
    NSTDM = cf['ox091096097081090080085086070']
    prec = 2 ** cf['ox091079086097108081078097078108083089']
    prec2 = 2 ** (2 * cf['ox091079086097108081078097078108083089'] - 1)
    prec3 = 2 ** (2 * cf['ox091079086097108081078097078108083089'])

    # get value
    dy_dr2_s  = res['fea.s.dx']
    dy_dr2_sr = res['fea.sr.dx']
    dy_dr2_g  = res['fea.g.dx']

    dy_dr2_s  = np.floor(dy_dr2_s  * prec3) / prec3
    dy_dr2_sr = np.floor(dy_dr2_sr * prec3) / prec3
    dy_dr2_g  = np.floor(dy_dr2_g  * prec3) / prec3

    dy_dr2 = dy_dr2_s + dy_dr2_sr + dy_dr2_g
    dy_dr2 = np.floor(dy_dr2 * prec) / prec
    
    test_range(dy_dr2_s  * prec, 'dy_dr2_s' )
    test_range(dy_dr2_sr * prec, 'dy_dr2_sr')
    test_range(dy_dr2_g  * prec, 'dy_dr2_g' )

    # cal fij
    rij = res['o_rij']
    rij = np.round(rij * prec) / prec
    fij1 = np.reshape(dy_dr2, [Na, -1, 1]) * np.reshape(rij, [Na, -1, 3]) * 2

    dy_drij = res['fea.r4.dx'][:,:,1:4]
    drij_dxyz = np.reshape(res['fea.sr.y'], [Na, -1, 1])
    fij2 = dy_drij * drij_dxyz
    fij2 = np.floor(fij2 * prec) / prec

    fij = fij1 + fij2

    nlist = intt(res['o_nlist']).reshape([Na, NI])
    fi = np.zeros([Na, 3])
    fijs = np.zeros([Na, 3])
    for ii in range(Na):
        for jj in range(NI):
            idx = nlist[ii, jj]
            if idx >= 0:
                fi[ii]  += fij[ii, jj]
                fi[idx] -= fij[ii, jj]
                fijs[ii] += fij[ii, jj]
    
    # round
    dy_dr2_s  = resort(dy_dr2_s  )
    dy_dr2_sr = resort(dy_dr2_sr )
    dy_dr2_g  = resort(dy_dr2_g  )
    dy_dr2    = resort(dy_dr2    )
    fi   = resort(fi  )
    fij1 = resort(fij1)
    fij2 = resort(fij2)
    fij  = resort(fij )
    fijs = resort(fijs)

    dy_dr2_s  = np.reshape(dy_dr2_s  ,[Na, -1])
    dy_dr2_sr = np.reshape(dy_dr2_sr ,[Na, -1])
    dy_dr2_g  = np.reshape(dy_dr2_g  ,[Na, -1])
    dy_dr2    = np.reshape(dy_dr2    ,[Na, -1])
    fij1      = np.reshape(fij1      ,[Na, -1])
    fij2      = np.reshape(fij2      ,[Na, -1])
    fij       = np.reshape(fij       ,[Na, -1])

    dy_dr2_s  = np.round(dy_dr2_s  * prec3)
    dy_dr2_sr = np.round(dy_dr2_sr * prec3)
    dy_dr2_g  = np.round(dy_dr2_g  * prec3)
    dy_dr2    = np.round(dy_dr2    * prec)
    fij1      = np.round(fij1      * prec2)
    fij2      = np.round(fij2      * prec)
    fij       = np.round(fij       * prec2)
    fi        = np.round(fi        * prec2)
    fijs      = np.round(fijs      * prec2)

    # save
    fio.save_txt3("./out/%s.txt"%('dy_dr2_s' ), dy_dr2_s , fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('dy_dr2_sr'), dy_dr2_sr, fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('dy_dr2_g' ), dy_dr2_g , fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('dy_dr2'   ), dy_dr2   , fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('fij1'     ), fij1     , fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('fij2'     ), fij2     , fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('fij'      ), fij      , fmt="%14d")
    fio.save_txt3("./out/%s.txt"%('fi'       ), fi       , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('fijs'     ), fijs     , fmt="%14d")

    test_range(fij, 'fij')
    test_range(fi, 'fi')
    test_range(fijs, 'fijs')


# save_feaNet
# =====================================================================


def save_feaNet(res):
    """
    fea.[s,sr,g].[x,y,dydx,dx]
    (Na, NI, -1)
    """
    print("INFO: Save FEANET")
    cf = read_cf()
    NSTDM = cf['ox091096097081090080085086070']
    Na = cf['ox091110']
    NI = cf['ox091086']
    Na_S = intt(Na * NSTDM)

    prec = 2 ** (cf['ox091079086097108083082078'] - 1)
    prec2 = 2 ** cf['ox091079086097108083082078108101']
    prec3 = 2 ** cf['ox091079086097108081078097078108083089']
    prec4 = 2 ** cf['ox091079086097108083082078108083089080068075086097072071067097072']

    # get data
    res2 = {}
    res2.update(res)
    for var in 's,sr,g'.split(','):
        for xy in 'x,y,dy,dydx,dx'.split(','):
            name = "fea.%s.%s"%(var, xy)
            res2[name] = res2[name].reshape([Na, -1])
    
    u       = res2['fea.g.x']
    s       = res2['fea.s.y']
    sr      = res2['fea.sr.y']
    ds_dr2  = res2['fea.s.dydx']
    dsr_dr2 = res2['fea.sr.dydx']
    G       = res2['fea.g.y']
    dG_dr2  = res2['fea.g.dydx']
    dy_dG   = res2['fea.g.dy']

    dy_dG = resort(dy_dG)

    # round
    v1_u       = np.round(u       * prec2)
    v1_s       = np.round(s       * prec4)
    v1_sr      = np.round(sr      * prec4)
    v1_ds_dr2  = np.round(ds_dr2  * prec4)
    v1_dsr_dr2 = np.round(dsr_dr2 * prec4)
    v1_G       = np.round(G       * prec4)
    v1_dG_dr2  = np.round(dG_dr2  * prec4)
    v1_dy_dG   = np.round(dy_dG   * prec3)

    # save
    fio.save_txt3("./out/%s.txt"%('u'      ), v1_u      , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('s'      ), v1_s      , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('sr'     ), v1_sr     , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('ds_dr2' ), v1_ds_dr2 , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('dsr_dr2'), v1_dsr_dr2, fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('G'      ), v1_G      , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('dG_dr2' ), v1_dG_dr2 , fmt="%8d")
    fio.save_txt3("./out/%s.txt"%('dy_dG'  ), v1_dy_dG  , fmt="%8d")

    # test range
    print("INFO: range of ds_dr2")
    test_range(v1_u, 'u')
    test_range(v1_s, 's')
    test_range(v1_sr, 'sr')
    test_range(v1_ds_dr2, 'ds_dr2')
    test_range(v1_dsr_dr2, 'dsr_dr2')
    test_range(v1_G, 'G')
    test_range(v1_dG_dr2, 'dG_dr2')
    test_range(v1_dy_dG, 'dy_dG')


# save_fea
# =====================================================================

def save_fea(res):
    print("INFO: Save FEA")
    cf = read_cf()
    Na = cf['ox091110']
    M1 = cf['ox090062']
    prec = 2 ** cf['ox091079086097108081078097078108083089']

    # get data
    r4  = res['fea.r4.y' ]
    dr4 = res['fea.r4.dx']
    gr  = res['fea.gr'   ]
    dgr = res['fea.dgr'  ]
    d   = res['fea.d'    ]
    dd  = res['fea.dd'   ]

    gr = resort(gr)
    dgr = resort(dgr)
    r4 = resort(r4)
    dr4 = resort(dr4)
    d  = resort(d)
    dd = resort(dd)
    
    r4  = r4 .reshape([Na, -1])
    dr4 = dr4.reshape([Na, -1])
    gr  = gr .reshape([Na, 4, M1])
    dgr = dgr.reshape([Na, 4, M1])
    d   = d  .reshape([Na, -1])
    dd  = dd .reshape([Na, -1])

    gr = np.transpose(gr, [0, 2, 1])
    dgr = np.transpose(dgr, [0, 2, 1])

    # round
    r4 = np.round(r4 * prec)
    dr4 = np.round(dr4 * prec)
    gr = np.round(gr * prec)
    dgr = np.round(dgr * prec)
    d  = np.round(d  * prec)
    dd = np.round(dd  * prec)

    # save
    np.savetxt("./out/%s.txt"%('r4'), r4, fmt="%8d")
    np.savetxt("./out/%s.txt"%('dr4'), dr4, fmt="%8d")
    fio.save_txt2("./out/%s.txt"%('gr'), gr, fmt="%8d")
    fio.save_txt2("./out/%s.txt"%('dgr'), dgr, fmt="%8d")
    fio.save_txt2("./out/%s.txt"%('d' ), d , fmt="%8d")
    fio.save_txt2("./out/%s.txt"%('dd' ), dd , fmt="%8d")
    # fio.save_txt3("./out/%s.txt"%('gr'), gr, fmt="%8d")
    # fio.save_txt3("./out/%s.txt"%('dgr'), dgr, fmt="%8d")
    # fio.save_txt3("./out/%s.txt"%('d' ), d , fmt="%8d")
    # fio.save_txt3("./out/%s.txt"%('dd' ), dd , fmt="%8d")

    # test range
    test_range(gr, 'gr', 4)
    test_range(dgr, 'dgr', 4)
    test_range(r4, 'r4', 4)
    test_range(dr4, 'dr4')
    test_range(d, 'd')
    test_range(dd, 'dd')


# save_fit
# =====================================================================


def save_fit(res):
    print("INFO: Save FIT")
    cf = read_cf()
    nlayer_fit = cf['ox123121110134114127108115118129']
    Na = cf['ox091110']
    NSTDM = cf['ox091096097081090080085086070']
    prec = 2 ** cf['ox091079086097108081078097078108083089']

    for ll in range(nlayer_fit):
        for var in 'wx,tanh,y'.split(','):
            for xy in 'y,dx'.split(','):
                name = "fit.%d.%s.%s"%(ll, var, xy)
                v = res[name]
                if (np.size(v) // (Na*NSTDM)) > 0:
                    v = v.reshape([Na*NSTDM, -1])
                v = np.round(v * prec)
                test_range(v, name)
                np.savetxt("./out/%s.txt"%name, v, fmt="%8d")

# save_out
# =====================================================================

def save_out(res):
    print("INFO: Save OUT")
    cf = read_cf()
    prec = 2 ** cf['ox091079086097108081078097078108083089']
    prec2 = 2 ** (2*cf['ox091079086097108081078097078108083089']-1)

    coord = res['t_coord']
    force = res['o_force']
    energy = res['o_energy']
    atom_energy = res['o_atom_energy']

    atom_energy = resort(atom_energy)

    force1 = np.round(force * prec)
    force2 = np.round(force * prec2)
    atom_energy = np.round(atom_energy * prec)

    print("INFO: energy \n", energy[0])

    coord = np.reshape(coord, [-1, 3])
    force = np.reshape(force1, [-1, 3])
    atom_energy = np.reshape(atom_energy, [-1, 1])
    np.savetxt("./out/coord.txt", coord)
    np.savetxt("./out/force.txt", force, fmt="%8d")
    np.savetxt("./out/force2.txt", force2, fmt="%8d")
    np.savetxt("./out/atom_energy.txt", atom_energy, fmt="%8d")

# save_in
# =====================================================================

def code_head(ncfg, nnet, nfea, ngra, nlst, natm):
    # check_st ip type cfg net fea gra lst atm check_ed check_ed2
    prec = 2**16

    stype = (1 if(ncfg>0) else 0) | (2 if(nnet>0) else 0) | (4 if(nfea>0) else 0) | \
            (8 if(ngra>0) else 0) | (16 if(nlst>0) else 0) | (32 if(natm>0) else 0)
    stype = hex(prec+stype)[-4:]

    check_ed = hex(prec + (int('f0f0', 16) ^ int('0001', 16) ^ int(stype, 16)))[-4:]

    ncfg = hex(prec+ncfg)[-4:]
    nnet = hex(prec+nnet)[-4:]
    nfea = hex(prec+nfea)[-4:]
    ngra = hex(prec+ngra)[-4:]
    nlst = hex(prec+nlst)[-4:]
    natm = hex(prec+natm)[-4:]

    head = "0f0f_%s_%s_%s_%s_%s_%s_%s_%s_0001_f0f0"%(check_ed, natm, nlst, ngra, nfea, nnet, ncfg, stype)
    return head.replace('_','')

def save_vcs_in(res):
    print("INFO: Save VCS OUT")
    cf = read_cf()
    Na = cf['ox091110']
    NI = cf['ox091086']
    NBIT_SPE_MAX = cf['ox091079086097108096093082108090078101']
    NBIT_LONG_DATA = cf['ox091079086097108089092091084108081078097078']
    NBIT_LONG_DATA_FL = cf['ox091079086097108089092091084108081078097078108083089080068075086097078081080073097070067086067097072']

    lst = res['o_nlist']
    atm = res['t_coord'].reshape([-1, 3])
    spe = res['t_type'].reshape([-1, 1])
    n = int(np.ceil(NI / 32))

    # lst
    lst = np.reshape(lst, [Na, -1])
    for ii in range(Na):
        v = lst[ii]
        v[v == -1] = ii
        lst[ii] = v
    lst = np.reshape(lst, [Na*n, -1])
    hsl = code_hb.arr2hex(lst, cf['ox091079086097108089096097108090078101'], nmerge=32*cf['ox091079086097108089096097108090078101'], signed=False, is_HLN0=True)

    # atm
    atm = np.round(atm * 2 ** NBIT_LONG_DATA_FL)
    hsx = code_hb.arr2hex(atm[:,0], NBIT_LONG_DATA, nfullzero=64, nmerge=NBIT_LONG_DATA, signed=True)
    hsy = code_hb.arr2hex(atm[:,1], NBIT_LONG_DATA, nfullzero=64, nmerge=NBIT_LONG_DATA, signed=True)
    hsz = code_hb.arr2hex(atm[:,2], NBIT_LONG_DATA, nfullzero=64, nmerge=NBIT_LONG_DATA, signed=True)
    hss = code_hb.arr2hex(spe, NBIT_SPE_MAX, nfullzero=64, nmerge=NBIT_SPE_MAX, signed=False)

    hsa = [hss[ii]+hsz[ii]+hsy[ii]+hsx[ii] for ii in range(Na)]
    hsa.append('')
    hsa_ = [(hsa[2*ii+1] + hsa[2*ii]) for ii in range((Na+1)//2)]

    nl = len(hsl) + len(hsa_)
    if nl < 64:
        print("#INFO: The number of nline should bigger than 64")
        hsa_.extend(['0'] * 64)

    # head
    head = code_head(0, 0, 0, 0, len(hsl), len(hsa_))

    # head + lst + atm
    s = [head]
    s.extend(hsl)
    s.extend(hsa_)
    np.savetxt('lmp2_hex.txt', s, fmt="%s")
