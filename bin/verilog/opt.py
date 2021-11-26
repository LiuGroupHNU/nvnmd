
import numpy as np

def mul2disp(x='', w=1, nbit=0):
    sign = 1 if w > 0 else 0
    ws = bin(int(np.abs(w)*(2**nbit))).replace('0b','')
    l = len(ws)
    wlist = []
    for ii in range(l):
        if ws[l-1-ii] == '1':
            wlist.append(nbit-ii)
    
    if x == '':
        return sign, [str(wi) for wi in wlist]
    else:
        s = []
        for wi in wlist:
            si = "(%s >>> %d)"%(x, wi) if wi >= 0 else  "(%s <<< %d)"%(x, -wi)
            s.append(si)
        if len(s) == 0:
            return ''
        else:
            if sign == 0:
                return '-' + ('-'.join(s))
            else:
                return '+'.join(s)


def cal_map_param(rc):
    rc2 = rc**2
    th =  np.log(1.5) / np.log(2)

    ln2_k1 = np.ceil(np.log(rc2/4) / np.log(2) -th)
    ln2_k2 = np.ceil(np.log(rc2*1) / np.log(2) -th)
    ln2_k3 = np.ceil(np.log(rc2*4) / np.log(2) -th)

    k1 = np.power(2, -ln2_k1)
    k2 = np.power(2, -ln2_k2)
    k3 = np.power(2, -ln2_k3)
    # b
    b1 = 0
    b2 = 0 #?
    b3 = 1.0 - rc2 * k3
    # xp
    x13, y13 = xp(k1, b1, k2, b2)
    b2 = y13 - x13 * k3
    while(True):
        x12, y12 = xp(k1, b1, k2, b2)
        x23, y23 = xp(k2, b2, k3, b3)
        dx = (x23 - x12) - rc2/2
        if (np.abs(dx) < 1e-6):
            break
        else:
            b2 += 0.01 * dx
    xps = np.array([[x12, y12], [x23, y23]])
    ks = np.array([k1, k2, k3])
    bs = np.array([b1, b2, b3])

    #-u2r-
    ks2 = 1 / (ks * rc2)
    xps2 = np.array([[y12, x12/rc2], [y23, x23/rc2]])
    b2_1 = 0
    b2_2 = xps2[0,1] - xps2[0,0] * ks2[1]
    b2_3 = 1.0 - 1.0 * ks2[2]
    bs2 = [b2_1, b2_2, b2_3]
    return xps, ks, bs, xps2, ks2, bs2

def map(xo, xps, ks, bs):
    x = xo + 0.0
    px, py = xps[0]
    k = ks[0]
    b = bs[0]
    I = xo <= px
    x[I] = xo[I] * k + b
    for ii in range(len(xps)):
        px, py = xps[ii]
        k = ks[ii+1]
        b = bs[ii+1]
        I = xo > px
        x[I] = xo[I] * k + b
    return x

def xp(k1, b1, k2, b2):
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x, y


def opt_too_small_w(w, th):
    N, M, K=  w.shape
    w2 = w + 0
    for kk in range(1, K):
        wij = w[:,:,kk]
        wij_past = w[:,:,kk-1]
        I = wij < th
        if np.sum(I) > 0:
            wij[I] = wij_past[I] - 1
            wij_past[I] -= 1
            w2[:,:,kk-1] = wij_past
            w2[:,:,kk] = wij
    return w2

def get_GRRG_idx(M1, M2):
    idxs = []
    for ii in range(M2):
        for jj in range(ii, M1):
            idxs.append(jj*M2 + ii)
    idxs = np.int32(idxs)
    return idxs

def get_opt_lst(lst, Nis):
    ntype  = len(Nis)
    Na, Ni = lst.shape
    st = 0
    idxs_list = np.zeros([Na, ntype])
    for tt in range(ntype):
        ni = Nis[tt]
        lst_i = lst[:,st:st+ni]
        idxs_list[:, tt] = np.sum(lst_i>-0.5, axis=1)
        st += ni

    idxs_list = np.int32(idxs_list)
    return idxs_list

def sort_lst(lst, lst2):
    I = np.argsort(-lst)
    I2 = np.argsort(-lst2)
    I3 = np.argsort(I2)
    I4 = I[I3]
    return I4

def get_opt_lst2(lst, lst2, Nis):
    ntype  = len(Nis)
    Na, Ni = lst.shape
    st = 0
    lst2 = lst2[:,:Ni]
    idxs_list = np.zeros([Na, Ni])
    for ii in range(Na):
        idxs_list[ii] = sort_lst(lst[ii], lst2[ii])
    idxs_list = np.int32(idxs_list)
    return idxs_list


def transpose_arr_by_lst(arr, Nis, NI, init_value, idxs_list):
    shape = arr.shape
    shape2 = np.int32(np.array(shape)) + 0
    shape2[1] = NI
    Na = shape[0]
    arr2 = np.zeros(shape2) + init_value
    # print(shape)

    ntype = len(Nis)
    st = 0
    idxs_st = np.zeros(Na, dtype=np.int32)
    print('###', arr.shape)
    for tt in range(ntype):
        ni = Nis[tt]
        arr_i = arr[:, st:st+ni]
        idxs_dx = idxs_list[:, tt]
        for ii in range(Na):
            idx_st = idxs_st[ii]
            idx_dx = idxs_dx[ii]
            arr2[ii, idx_st:idx_st+idx_dx] = arr_i[ii, 0:idx_dx]
            idxs_st[ii] += idx_dx
        st += ni
    return arr2

def transpose_arr_by_lst2(arr, Nis, NI, init_value, idxs_list):
    shape = arr.shape
    shape2 = np.int32(np.array(shape)) + 0
    shape2[1] = NI
    Na = shape[0]
    arr2 = np.zeros(shape2) + init_value

    for ii in range(Na):
        arr_i = arr[ii]
        arr2[ii] = arr_i[idxs_list[ii]]
    return arr2

def code_st(signals, times):
    signals = signals.replace(';', ' ')
    signals = signals.replace(',', ' ')
    signals = signals.split()
    stDic = dict(zip(signals, times))
    return stDic

def split_factor(n):
    """:SPlit n=a1*a2
    """
    q = int(np.floor(np.sqrt(n)))
    a1 = 1
    a2 = n
    for ii in range(q, 0, -1):
        if ((n // ii) * ii) == n:
            a1 = ii 
            a2 = n // ii
            break
    return a2, a1 # a2 > a1


def float2sem(v):
    e_127 = np.floor(np.log(np.abs(v)) / np.log(2))
    e_127 = 23 - e_127
    m_1 = v * np.power(2, e_127) # 1.m
    return np.int64(m_1), np.int64(e_127)

def double2sem(v):
    e_1023 = np.floor(np.log(np.abs(v)) / np.log(2))
    e_1023 = 52 - e_1023
    I = np.abs(v) < 1e-10
    e_1023[I] = 0
    m_1 = v * np.power(2, e_1023) # 1.m
    return np.int64(m_1), np.int64(e_1023)

def disp_double2sem(v):
    v = np.array(v)
    m, e = double2sem(v)
    n = len(m)
    s = [str(m[ii]) for ii in range(n)]
    s = ', '.join(s)
    print('m = [ %s ]'%s)
    s = [str(e[ii]) for ii in range(n)]
    s = ', '.join(s)
    print('e = [ %s ]'%s)

def sem2double(m, e):
    m = np.int64(m)
    e = np.int64(e)
    v = np.float64(m)
    v = v * np.power(0.5, e)
    return v


def quantify(x, nbit=14, is_round=False):
    prec = 2**nbit
    if is_round:
        return np.round(x * prec) / prec
    else:
        return np.floor(x * prec) / prec






