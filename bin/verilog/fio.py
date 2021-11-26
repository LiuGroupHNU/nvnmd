


import os
import json
import numpy as np
import scipy.io as sio
from ase.io import write, read

from code_hb import get_lines

def test_arr(name, arr):
    print(name, ':', np.min(arr), np.max(arr), np.mean(arr), np.std(arr))
    arr = np.abs(arr)
    print(name, ' abs:', np.min(arr), np.max(arr), np.mean(arr), np.std(arr))


def print_arr(arr):
    sh = arr.shape
    nrow, ncol = sh[0], sh[1]

    s = ''.join(["%5d "%(jj) for jj in range(ncol)])
    print('%5s '%('###')+s)
    for ii in range(nrow):
        s = ""
        for jj in range(ncol):
            s += "%5d "%arr[ii, jj]
        print('%5d:'%(ii)+s)


def load_npy(fn):
    if os.path.exists(fn):
        return np.load(fn, allow_pickle=True)
    else:
        print("ERROR: mismatch file \n", fn)
        return None

def load_txt(fn):
    if os.path.exists(fn):
        return np.loadtxt(fn)
    else:
        print("ERROR: mismatch file \n", fn)
        return None

def load_json(fn):
    if os.path.exists(fn):
        fr = open(fn, 'r')
        jdata = json.load(fr)
        fr.close()
        return jdata
    else:
        print("ERROR: mismatch file \n", fn)
        return None

def load_atoms(dataPath, mod='atm', cf={}):
    """
    从文件夹dataPath下,根据模式，读入相应的文件，读取得到原子信息
    模式分为:atm, hex, 和vel
        1.atm 读取atoms.xsf和atoms_past.xsf
        2.hex 读取atoms.xsf和atoms_hex.xsf
        3.vel 读取atom.xsf和atoms_vel.xsf
    #Output:
        atoms, crd_diff
    """

    if mod == 'atm':
        atoms = read(dataPath+'/atoms.xsf')
        crd = atoms.get_positions()

        atoms_past = read(dataPath+'/atoms_past.xsf')
        crd_past = atoms_past.get_positions()
        crd_diff = crd - crd_past
    
    if mod == 'vel':
        atoms = read(dataPath+'/atoms.xsf')
        crd = atoms.get_positions()

        atoms_vel = read(dataPath+'/atoms_vel.xsf')
        crd_diff = atoms_vel.get_positions()
    
    # if mod == 'hex':
    #     hs = get_lines(dataPath+'/atoms_hex.txt')
    #     spe, crd, crd_diff = process_decode_spe_crd_crd_diff(hs, cf)

    return atoms, crd_diff


def save_atoms(dataPath, atoms, crd_diffs, mod='atm', cf={}):
    """
    将原子信息以一定格式保存在dataPath目录下
    """

    write(dataPath+'/atoms.xsf', atoms)

    if mod == 'atm':
        crd_pasts = atoms.get_positions() - crd_diffs
        atoms.set_positions(crd_pasts)
        write(dataPath+'/atoms_past.xsf')
    
    if mod == 'vel':
        atoms.set_positions(crd_diffs)
        write(dataPath+'/atoms_vel.xsf')

    # if mod == 'hex':
    #     hs = process_code_spe_crd_crd_diff(atoms, crd_diffs, cf)
    #     np.savetxt(dataPath+'/atoms_hex.txt', hs, fmt="%s")


def save_dic_txt(dataPath, arr, fmt):
    """
    将组合的arr写在一个文件
    """
    np.savetxt(dataPath, arr, fmt=fmt)

def save_txt2(dataPath, arr, fmt):
    sh = arr.shape
    nd = len(sh)
    if nd > 2:
        arr = np.reshape(arr, [sh[0], -1])
    save_dic_txt(dataPath, arr, fmt)

def save_txt3(dataPath, arr, fmt):
    """
    可以最多保存3维数组为文本
    """
    sh = arr.shape
    nd = len(sh)
    if nd < 3:
        save_dic_txt(dataPath, arr, fmt)
    else:
        # sh = [i for i in sh]
        # sh.append(1)
        # sh.append(1)
        # sh.append(1)
        # sh = sh[0:3]

        # arr = arr.reshape(sh)
        lines = []
        for ii in range(sh[0]):
            s = ''
            for jj in range(sh[1]):
                if (nd == 3) or (jj % 10 == 0):
                    s += " [%5d:%5d] "%(ii, jj)
                for kk in range(sh[2]):
                    s += fmt%arr[ii, jj, kk]
            lines.append(s+'\n')
        fw = open(dataPath, 'w')
        fw.writelines(lines)
        fw.close()

    