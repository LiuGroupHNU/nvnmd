
import sys
import copy
import numpy as np


###                        Arr <-> Bin <-> Hex
#======================================================================

def val2bin(v, NBIT):
    """
    # 功能
    正整数v编码成NBIT二进制字符串
    """
    b = bin(int(v)).replace('0b','')
    b = ('0'*NBIT) + b
    return b[-NBIT:]

def arr2bin(arr, nbit, signed=False):
    '''
    # 功能
    数组arr中每个数都编码成nbit二进制字符串，signed配置是否输入为有符号整数
    '''
    arr = np.int64(np.reshape(arr, [-1]))
    n = len(arr)
    prec = np.int64(np.power(2, nbit))
    if signed:
        pmax = prec // 2 - 1
        pmin = -pmax
    else:
        pmax = prec - 1
        pmin = 0

    li = []
    s0 = '0' * nbit
    for ii in range(n):
        v = arr[ii]
        v = np.min([v, pmax])
        v = np.max([v, pmin])
        s = bin(int(v) + prec)
        s = s[-nbit:]
        li.append(s)
        #**test
        if '-' in s:
            print(ii, n, arr[ii], prec)
    return li

def bin2hex(str_bin):
    """
    # 功能
    二进制字符串转十六进制字符串
    """
    str_bin = str_bin.replace('0b','')
    str_bin = str_bin.replace('b','')

    # 4转1，不过就高位补0
    l = len(str_bin)
    n = int(np.ceil(l / 4))
    dl = int(n * 4 - l)
    s = ('0'*dl) + str_bin

    # 4转1
    slist = []
    for ii in range(n):
        si = hex(int(s[ii*4:(ii+1)*4], 2)).replace('0x','')
        slist.append(si)
    str_hex = ''.join(slist)
    return str_hex

def hex2bin(str_hex):
    """
    # 功能
    十六进制转二进制
    """

    str_hex = str_hex.replace('0x','')
    l = len(str_hex)
    n = l * 4
    slist = []
    # 1转4
    for ii in range(l):
        si = '0000' + bin(int(str_hex[ii], 16)).replace('0b','')
        slist.append(si[-4:])
    str_bin = ''.join(slist)
    return str_bin

def arr2hex(arr, nbit, nmerge=None, nfullzero=None, signed=False, is_HLN0=False):
    '''
    # 功能
    将数组编码成16进制数

    首先转换数值成2进制
    然后将每{nmerge}位二进制数变成1个16进制数

    # 参数
    arr 数值数组
    nbit 每个数转换成多少bit
    nmerge  每（nfullzero-nmerge)个1'b0和nmerge个bit的数据转换成1个16进制数
            当 nfullzero<nmerge， nfullzero = nmerge
    nfullzero 
    is_HLN0 True: 后面的数值转换成的16进制会在高位
            False: 在低位
    signed True: 按无符号数进行转换
           False: 按有符号数进行转换
    '''
    # 计算输入
    nmerge = nbit if nmerge == None else nmerge
    nfullzero = 1 if nfullzero == None else nfullzero
    nfullbit = int(np.ceil(nmerge / nfullzero) * nfullzero)
    nz = int(np.ceil(nfullbit / 4))
    nm = int(np.ceil(nmerge / 4))
    ds = '0'*(nz - nm)

    # is_HLN0功能
    #**即使用反序来实现
    arr2 = copy.copy(arr) + 0
    if is_HLN0:
        sh_arr = arr2.shape
        nrow, ncol = sh_arr[0], sh_arr[1]
        arr2[:,:] = arr2[:,::-1]
        # for ii in range(int(np.floor(ncol/2))):
        #     tmp = arr2[:, ncol-1-ii] + 0
        #     arr2[:, ncol-1-ii] = arr2[:, ii] + 0
        #     arr2[:, ii] = tmp + 0
    
    # arr2bin
    li = arr2bin(arr2, nbit, signed=signed)
    s = ''.join(li)
    s = s.replace('0b','')
    s = s.replace('b','')
    
    # bin2hex
    n = len(s)
    li = []
    for ii in range(n // nmerge):
        v = bin2hex(s[nmerge*ii:nmerge*(ii+1)])
        v = ds + v
        li.append(v)
    return li

def arr2hexs(arrs, nbits, signeds):
    """
    # 功能
    arrs为链表，包括多个arr
    数组arr中每个数都编码成nbit二进制字符串，signed配置是否输入为有符号整数
    然后二进制转16进制
    """
    bs = []
    for ii in range(len(arrs)):
        v = arr2bin(arrs[ii], nbits[ii], signed=signeds[ii])
        bs.append(v)
    nrow = len(arrs[0])
    ss = []
    for ir in range(nrow):
        s = [bs[ii][ir] for ii in range(len(bs))]
        # print(s)
        s = ''.join(s)
        s = bin2hex(s)
        ss.append(s)
    return ss

###                              Reformat FileList
#======================================================================


def get_lines(fn):
    """
    # 功能
    读取文本文件，获得lines
    """
    fr = open(fn)
    lines = fr.readlines()
    fr.close()

    lines = [line.replace('\n','') for line in lines]
    return lines

def merge_hex_files(fnList, nbitList, fout):
    """
    # 功能
    合并保存有16进制字符串的文本文件
    # 参数
    fnList: 文件名链表
    nbitList: 文件中一行字符串设有nbit个二进制数
    fout: 输出文件名
    """
    linesList = []
    nfn = len(fnList)
    for ii in range(nfn):
        fn = fnList[ii]
        fr = open(fn, 'r')
        lines = fr.readlines()
        fr.close()
        lines = [line.replace('\n', '') for line in lines]
        linesList.append(lines)
        #test
        print("%s file nbit_one_row is: %d, and decode as %d"%(fn, len(lines[0]*4), nbitList[ii]))
    linexList = merge_hex_strList(linesList, nbitList)
    np.savetxt(fout, linexList, fmt='%s')

###                              Reformat StrList
#======================================================================


def merge_hex_strList(linesList, nbitList):
    """
    # 功能
    有多个lines在linesList,
    有对应的nbit在nbitList，
    lines保存有nbit个二进制数，
    新的lines(i)是将所有lines(i)按各自长度nbit拼接起来
    """
    nline = len(linesList[0])
    ncol = np.sum(nbitList)
    print("the file has %d row"%(nline))
    print("the file has %d col"%(ncol))
    linexList = []

    # 合并
    for ii in range(nline):
        linex = []
        for jj in range(len(linesList)):
            str_bin = hex2bin(linesList[jj][ii])
            dl = nbitList[jj] - len(str_bin)
            if dl <= 0: 
                linex.append(str_bin[-nbitList[jj]:])
            else: 
                linex.append(('0'*dl) + str_bin)
        str_bin = ''.join(linex)
        str_hex = bin2hex(str_bin)
        linexList.append(str_hex)
    return linexList

def spilit_hex_strList(linesList, nbitList, nbit_shift=None):
    """
    # 功能
    与函数merge_hex_strList的操作相反
    """
    nline = len(linesList)
    nhex = len(linesList[0])
    ncol = int(nhex * 4)
    ndata = len(nbitList)

    if nbit_shift == None:
        if ncol >= np.sum(nbitList):
            nbit_shift = ncol - np.sum(nbitList)
        else:
            print("每行需要的bit数大于实际值")

    print("the file has %d row"%(nline))
    print("the file has %d col"%(ncol))

    dataList = []
    for ii in range(ndata):
        dataList.append([])
    
    for ii in range(nline):
        str_bin = hex2bin(linesList[ii])

        st = nbit_shift
        for jj in range(ndata):
            nbit = nbitList[jj]
            str_hex = bin2hex(str_bin[st:st+nbit])
            dataList[jj].append(str_hex)
            st += nbit
    return dataList
