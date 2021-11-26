

import os
import numpy as np
from ase.io import read 
import matplotlib.pyplot as plt

# * ------------------------------------------------------------------- 
# check and cmd
# * ------------------------------------------------------------------- 


def cmd_find(par, v, dic):
    print("#INFO: cmd_find")
    print("cmd,err,key1,key2,...,keyn")

    t = 0
    err = par[t]; t+= 1
    key = par[t:]

    err = float(err)
    key = [float(k) for k in key]
    key = np.array(key)

    nd = len(key)
    dd = v[:,0:nd] - key
    idx = np.sum((np.abs(dd) <= err), axis=1) == nd

    print("key:", key)
    print("idx:")
    print(np.where(idx))
    print("error:")
    print(dd[idx])

    return v, dic

def decode_check_cmd(cmd_par, v, dic={}):
    """命令和参数用","分割,第一个参数是名字
    """
    cmd_par_s = cmd_par.split(",")
    cmd = cmd_par_s[0]
    par = cmd_par_s[1:]
    print("\n#",cmd, ":", par)
    #
    if cmd == 'min':
        v = np.min(v)
    #
    if cmd == 'max':
        v = np.max(v)
    #
    if cmd == 'len':
        v = len(v)
    #
    if cmd == 'fla':
        v = np.reshape(v, [-1])
        for ii in range(len(v)):
            print(ii, v[ii])
    #
    if cmd == 'int':
        v = np.int32(v)
    #
    if cmd == 'idx':
        if len(par) == 1:
            idx = int(par[0])
            v = v[idx]
        if len(par) == 2:
            if par[0] == 'n':
                idx = int(par[1])
                v = v[:,idx]
            else:
                idx = int(par[0])
                idx2 = int(par[1])
                v = v[idx, idx2]
    #
    if cmd == 'T':
        v = np.transpose(v)
    #
    if cmd == 'k':
        v = v *float(par[0])
    #
    if cmd == 'find':
        v = cmd_find(par, v, dic)
    #
    if cmd == 'reshape':
        nr = int(par[0])
        nc = int(par[1])
        v = np.reshape(v, [nr, nc])
    #
    if cmd == "print":
        print(v)
    #
    if cmd == "print2d":
        for ii in range(v.shape[0]):
            s = []
            for jj in range(v.shape[1]):
                s.append(str(v[ii, jj]))
            s = ','.join(s)
            print(s)
    #
    if cmd == "help":
        print("min")
        print("max")
        print("len")
        print("fla")
        print("find,err,key1,key2,...keyn")
        print("idx: return idx-th row")
        print("int")
        print("k,k_fact: v*k_fact")
        print("T transpose")
        print("reshape,nr,nc")
        print("print")
        print("print2d")
    return v, dic

def check(argvs):
    print("#INFO: check")
    print("fn key:cmd1,par1:cmd2,pars:...:cmdn,parn")
    print("defulat key is val")

    fn = argvs[0]
    argvs = argvs[1:]

    #
    if ".npy" in fn:
        dat = np.load(fn, allow_pickle=True)[0]
        keys = dat.keys()
        if len(argvs) == 0:
            for key in keys:
                print(key)
        
    elif ".txt" in fn:
        keys = ['val']
        dat = np.loadtxt(fn)
        dat = {'val':dat}
    
    # cmd
    if len(argvs) > 0:
        print(keys)
        for argv in argvs:
            pars = argv.split(":")
            key = pars[0]
            cmd_pars = pars[1:]

            val = dat[key]
            dic = {}
            print("# ", key)
            print(argv, cmd_pars)
            for cmd_par in cmd_pars:
                val, dic = decode_check_cmd(cmd_par, val, dic)



# * ------------------------------------------------------------------- 
# test force
# * ------------------------------------------------------------------- 

def read_frc_hex(fn, natom):
    lines = open(fn, 'r').readlines()
    lines = lines[1:]
    lines = [line.replace('\n','') for line in lines]

    ct = 0
    data = []
    nl = len(lines)
    for ii in range(nl):
        if ct >= natom*3: break
        line = lines[ii]
        for jj in range(15):
            if ct >= natom*3: break
            ss = line[-8:]
            line = line[:-8]
            v = int(ss, 16)
            data.append(v)
            ct += 1
    data = np.array(data)
    data[data>=2**31] -= 2**32
    data = data.reshape([-1, 3])
    return data

def read_lst_hex(fn, natom):
    lines = open(fn, 'r').readlines()
    lines = lines[1:]
    lines = [line.replace('\n','') for line in lines]

    NI = 128

    ct = 0
    data = []
    nl = len(lines)
    for ii in range(nl):
        if ct >= natom*NI: break
        line = lines[ii]
        for jj in range(32):
            if ct >= natom*NI: break
            ss = line[-4:]
            line = line[:-4]
            v = int(ss, 16)
            data.append(v)
            ct += 1
    data = np.array(data)
    data = np.int32(data)
    data = data.reshape([-1, NI])
    return data

def read_data(atm_fn, idx_fn, fi_fn, tag_fn, frc_fn, lst_fn):
    frc = np.loadtxt(fi_fn) / 2**25
    tag = np.int32(np.loadtxt(tag_fn))
    frc2 = read_frc_hex(frc_fn, len(tag)) / 2**25
    atoms = read(atm_fn)
    lst = read_lst_hex(lst_fn, len(frc))

    if os.path.exists(idx_fn):
        idx = np.int32(np.loadtxt(idx_fn))
    else:
        idx = np.int32(np.arange(len(frc)))
    return idx, tag, frc, frc2, atoms, lst


def plot_yx(f, f2):
    plt.plot(f[:,0], f2[:,0], '.b', label='Fx', alpha=0.5)
    plt.plot(f[:,1], f2[:,1], '.g', label='Fy', alpha=0.5)
    plt.plot(f[:,2], f2[:,2], '.r', label='Fz', alpha=0.5)
    plt.plot(f.reshape([-1]), f.reshape([-1]), '-k')
    plt.title('force y=x')
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

def plot_yx_spe(f, f2, spe):
    types = np.unique(spe)
    nt = len(types)
    for ii in range(nt):
        idx = spe == types[ii]
        fi = f[idx].reshape([-1])
        f2i = f2[idx].reshape([-1])
        plt.plot(fi, f2i, '.', label="%d"%(types[ii]))

    idx = test(f, f2)
    fi = f[idx].reshape([-1])
    f2i = f2[idx].reshape([-1])
    plt.plot(fi, f2i, 'xr', label="error", alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title('force with spe')
    plt.show()
    plt.close()

def test(f, f2):
    df = np.sqrt(np.sum(np.power(f - f2, 2), axis=1))
    idx = df > 0.2
    return idx

def test_lst(f, f2, lst, tag):
    print("#INFO: test error atomic force")
    lst = tag[lst]-1
    # print(lst)

    idx = test(f, f2)
    lst_e = idx[lst]
    lst_e = np.sum(lst_e, axis=1)
    idx_e = lst_e == 128
    print("idx")
    print(np.where(idx_e))

def test_force(argvs):
    print("#INFO: test_force")

    if len(argvs) == 0:
        print("test_force atom.xsf idx.txt fi.txt tag.txt frc_hex.txt lst_hex.txt")
        print("atom.xsf: atoms structure file")
        print("idx.txt: output by \"dpvcs debug\" command")
        print("fi.txt: output by \"dpvcs debug\" command")
        print("tag.txt: output by debug of lammps, the tag of atoms")
        print("frc_hex.txt: output by debug of lammps, hex format, return by FPGA")
        print("lst_hex.txt: output by debug of lammps, hex format, neighborlist, the data for sending to FPGA")

    t = 0
    atm_fn = argvs[t]; t += 1
    idx_fn = argvs[t]; t += 1
    fi_fn = argvs[t]; t += 1
    tag_fn = argvs[t]; t += 1
    frc_fn = argvs[t]; t += 1
    lst_fn = argvs[t]; t += 1

    idx, tag, frc, frc2, atoms, lst = read_data(atm_fn, idx_fn, fi_fn, tag_fn, frc_fn, lst_fn)
    spe = atoms.get_atomic_numbers()

    frc = frc[idx]
    frc3 = frc*0
    for ii in range(len(tag)):
        frc3[tag[ii]-1] += frc2[ii]
    
    plot_yx(frc, frc3)
    plot_yx_spe(frc, frc3, spe)
    test_lst(frc, frc3, lst, tag)
















