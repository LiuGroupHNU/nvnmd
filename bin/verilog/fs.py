
import re
import copy
import numpy as np
import scipy.io as sio

## FILE AND LINES
#================

def get_writer(fn):
    fw = open(fn, 'w')
    fw.close()
    def w(s):
        fw = open(fn, 'a')
        fw.write(s + '\n')
        fw.close()
    return w
    
def read_fn(fn):
    fr = open(fn, 'r', encoding='utf-8')
    lines = fr.readlines()
    fr.close()
    return lines

def replace(dic, lines):
    """:Use the dic and command in line to generate a new line
    """
    CODE_ENTER = 'CODE_ENTER'
    lines2 = []
    ii = 0
    n = len(lines)
    while(ii < n):
        line = lines[ii]
        ii += 1

        #empty line
        if len(line.split()) == 0:
            lines2.append(line)
            continue
        
        #block comment line
        if '/*' in line:
            lines2.append(line)
            while("*/" not in lines[ii-1]):
                lines2.append(lines[ii])
                ii += 1
            continue

        #print message
        if 'PRINT(' in line:
             a = re.search('PRINT(.*)', line)
             st, ed = a.span()
             line2 = line[st:ed-1]
             line2 = expressIdx2value(line2, dic)
             print(line2)
             continue

        #comment line
        if line.split()[0].startswith('//'):
            lines2.append(line)
            continue
        
        #continuation character
        if '//+++' in line:
            while('//---' not in lines[ii-1]):
                line += lines[ii]
                ii += 1
            line = line.replace('//+++', '')
            line = line.replace('//---', '')
            line = line.replace('\n', CODE_ENTER)
        
        
        #command line
        a = re.search('<[^=].*>', line)
        if a != None:
            st, ed = a.span()[0], a.span()[1]
            #test the '()' and '{}'
            n1 = np.sum([c == '(' for c in line[st:ed]])
            n2 = np.sum([c == ')' for c in line[st:ed]])
            n3 = np.sum([c == '{' for c in line[st:ed]])
            n4 = np.sum([c == '}' for c in line[st:ed]])
            if (n1 != n2) or (n3 != n4):
                print('<><><><>Get fail fmt: (n1 != n2) or (n3 != n4)')
                print('@line:', line.replace(CODE_ENTER, '\n'))
                print('@(){\} n1, n2, n3, n4:', n1, n2, n3, n4)
                exit()
            
            #s0: prefix
            #s1: <*>
            #s2: postfix
            s0 = line[:st]
            s1 = line[st+1:ed-1]
            s2 = line[ed:]

            #run command
            fmt = s1
            fmt = fs_if(fmt, dic)
            fmt = fs_key(fmt, dic)
            fmt = fs_for(fmt, dic)
            fmt = fs_arr(fmt, dic)
            fmt, dic = fs_def(fmt, dic)
            fmt = expressIdx2value(fmt, dic)

            line = s0 + fmt + s2
        
        line = line.replace(CODE_ENTER, '\n')
        lines2.append(line)

    return lines2

## FS: FORMAT_STRING
#===================

def express2value(fmt, dic):
    ''':eval the value of {fmt}
    @fmt: the format string, such as 'a+b+1'
    @dic: the dictory includes the value of key in {fmt}
          such as the dic={'a':2, 'b':3}, return value is 6
    '''
    # find the variable_name (string)
    a = re.findall('[a-zA-Z0-9_]+', fmt)
    #sort by length
    a = np.array(a)
    al = [len(ai) for ai in a]
    al = np.array(al)
    idx = np.argsort(-al)
    a = a[idx]
    # replace the variable_name(string) to variable_value(string)
    for key in a:
        if key in dic.keys():
            fmt = fmt.replace(key, str(dic[key]))
    # find unknown variables
    a = re.findall('[a-zA-Z]+', fmt)
    a2 = []
    for ai in a:
        if ai not in ['and','or','not', 'e']:
            a2.append(ai)
    if len(a2) == 0: 
        return eval(fmt)
    else: 
        return fmt

def expressIdx2value(fmt, dic, ijk=None, vijk=None):
    ''':find '[*]' and replace the variable_name to value from dic
    '''
    #find '[*]' or '[*:*]'
    a = re.findall('\[[^[\]]+\]', fmt)
    # a = re.findall('\[[^[]+\]', fmt)
    # a = re.findall('\[[a-zA-Z0-9_+\-\*/ :\(\)]+\]', fmt)
    #dic
    dic2 = {}
    if ijk != None and vijk != None:
        dic2[ijk] = vijk
    dic2.update(dic)
    #sort by length
    a = np.array(a)
    al = [len(ai) for ai in a]
    al = np.array(al)
    idx = np.argsort(-al)
    a = a[idx]
    #replace
    for key in a:
        #'[*]' to *
        key2 = key.replace('[','').replace(']','')
        #'[*:*]' to * and *
        splt_key = re.findall('[+-]*:', key2)
        splt_key = splt_key[0] if len(splt_key) > 0 else ':'
        keys2 = key2.split(splt_key)
        p = True
        vlist = []
        for k in keys2:
            v = express2value(k, dic2)
            vlist.append(str(v))
            if type(v) == str:
                p = False
        v = splt_key.join(vlist)
        if (not p) or (':' in v):
            fmt = fmt.replace(key, '['+v+']')
        else:
            fmt = fmt.replace(key, str(v))
    return fmt

## FSX_LOOP

def fsx_get_loop_config(fmt, dic):
    '''
    i=NMIN:NMAX,str_concat
    the NMIN and NMAX is in the dic
    '''
    pars = fmt.split()
    sidx = pars[0]
    idx_pars = sidx.split('=')
    ijk = idx_pars[0]
    srange = idx_pars[1].split(':')
    while '' in srange:
        srange.remove('')
    if len(srange) == 2:
        smin, smax = srange
    else:
        print("<><><><>Get failure wrong config")
        print("@fun:", "fsx_get_loop_config")
        print("@fmt:", fmt)
        exit()
    # print(smin, smax)
    nmin = express2value(smin, dic)
    nmax = express2value(smax, dic)
    if str(nmin).isdigit() and str(nmax).isdigit():
        nmin = int(nmin)
        nmax = int(nmax)
    else:
        print("<><><><>ERROR in FUN(fsx_get_loop_config):")
        print('@nmin, nmax:', nmin, nmax)
        print('@fmt:', fmt)
        exit()
    if len(pars) == 1: str_concat = ""
    elif len(pars) == 2: str_concat = pars[1]
    return ijk, nmin, nmax, str_concat

def fsx_get_bracket_position(line, brack='{', nbrack='}', st=0):
    """:find the position of brackets
    """
    num_b = 0
    num_nb = 0 
    b_st = 0
    b_ed = 0
    for ii in range(st, len(line)):
        if line[ii] == brack:
            if num_b == 0:
                b_st = ii
            num_b += 1
        if line[ii] == nbrack:
            num_nb += 1
        if (num_b > 0) and (num_nb == num_b):
            b_ed = ii
            break
    return b_st, b_ed

def fs_for(fmt, dic):
    """
    :use the 'FOR(i=1:N ;){}' format to generate the repetitive verilog code
    @fmt: the format string to generate verilog code
    @dic: a dictory that includes the parameter in the {fmt} 
    """
    a = re.search('FOR\([^{}]+\)', fmt)
    # a = re.search('FOR[(0-9a-zA-Z=:,+\-\*/\\\ )]+', fmt)
    if a == None:
        fmt = fmt.replace('\\n', '\n')
        return fmt
    else:
        # find the head of loop format
        l = len(fmt)
        st, ed = a.span()[0], a.span()[1]
        sidx = fmt[st+4:ed-1] # '*' in 'FOR(*)'
        # get the loop information from the head
        ijk, nmin, nmax, str_concat = fsx_get_loop_config(sidx, dic)
        # find the string which need to be repeat generate
        stb, edb = fsx_get_bracket_position(fmt, '{', '}', ed)

        # s0: prefix
        # s1: loop setting
        # s2: {*}
        # s3: postfix
        s0 = fmt[:st]
        s1 = fmt[st:ed]
        s2 = fmt[stb+1:edb]
        s3 = fmt[edb+1:]

        # print('s0',s0)
        # print('s1',s1)
        # print('s2',s2)
        # print('s3',s3)

        # generate
        lineList = []
        dx = 1 if nmin <= nmax else -1
        dic2 = {}
        dic2.update(dic)
        for ii in range(nmin, nmax+dx, dx):
            dic2[ijk] = ii
            line = expressIdx2value(s2, dic2, ijk, ii)
            lineList.append(fs_for(line, dic2))
        s = s0 + (str_concat.join(lineList)) + s3
        s = fs_for(s, dic)
        s = s.replace('\\n', '\n')
        return s

def fs_if(fmt, dic):
    """
    :use the 'IF(p){line}' format to generate the verilog code
    {p} is judgment formula, and {line} is the content
    if {p} == True, return {line}; else return ''
    @fmt: the format string to generate verilog code
    @dic: a dictory that includes the parameter in the {fmt} 
    """
    # a = re.search('IF\([(0-9a-zA-Z_=+\-\*/\\><) ]+\)', fmt)
    a = re.search('IF\([^{}]+\)', fmt)
    if a != None:
        l = len(fmt)
        st, ed = a.span()[0], a.span()[1]
        sidx = fmt[st+3:ed-1] # 'IF(' and ')'
        stb, edb = fsx_get_bracket_position(fmt, '{', '}', ed)
        # s0: prefix
        # s1: judgment formula
        # s2: {*}
        # s3: postfix
        s0 = fmt[:st]
        s1 = fmt[st:ed]
        s2 = fmt[stb+1:edb]
        s3 = fmt[edb+1:]

        # print('s0',s0)
        # print('s1',s1)
        # print('s2',s2)
        # print('s3',s3)

        # generate
        p = express2value(sidx, dic)
        # print(sidx, p, s2)
        if p:
            s2 = expressIdx2value(s2, dic)
            fmt = s0 + s2 + s3
        else:
            fmt = s0 + s3
        return fs_if(fmt, dic)
    else:
        return fmt

def fs_key(fmt, dic):
    """:replace the key string to the value string from dic
    #-input
    @fmt: the string with <key_string> format
    @dic: is a key-value dictionary, containing key_string-value_string pair
    """
    a = re.search('[a-zA-Z0-9_]+', fmt)
    if a != None:
        l = len(fmt)
        st, ed = a.span()[0], a.span()[1]
        if st == 0 and ed == l:
            key = fmt
            if key in dic.keys():
                fmt = str(dic[key])
    return fmt

def fs_arr(fmt, dic):
    """:replace the string to the value of the arry
    the format is ARR(arr, [idx1, idx2, idx3])
    """
    while (True):
        a = re.search('ARR\([^()]+\)', fmt)
        if a != None:
            l = len(fmt)
            st, ed = a.span()[0], a.span()[1]
            sidx = fmt[st+4:ed-1] # 'ARR(' and ')'
            # s0: prefix
            # s1: command
            # s2: postfix
            s0 = fmt[:st]
            s1 = fmt[st:ed]
            s2 = fmt[ed:]
            # print(s0, '#', s1, '#', s2, '\n')

            # generate
            pars = sidx.split(',')
            arr_name = pars[0]
            idxs = pars[1:]
            arr = dic[arr_name]
            # print(arr_name, '=>', arr.shape, '=>', idxs)
            if len(idxs) == 0:
                v = "%d"%(arr)
            else:
                d = arr
                for idx in idxs:
                    d = d[int(idx)-1]
                v = "%d"%(d)
            fmt = s0 + v + s2
        else:
            break
    return fmt

def fs_def(fmt, dic):
    """:add the key-value pair to the dictionary
    the format is DEF(key,value_express)
    """
    while (True):
        a = re.search('DEF\([^()]+\)', fmt)
        if a != None:
            l = len(fmt)
            st, ed = a.span()[0], a.span()[1]
            sidx = fmt[st+4:ed-1] # 'DEF(' and ')'
            # s0: prefix
            # s1: command
            # s2: postfix
            s0 = fmt[:st]
            s1 = fmt[st:ed]
            s2 = fmt[ed:]

            # generate
            pars = sidx.split(',')
            key = pars[0]
            value = pars[1]
            dic[key] = express2value(value, dic)
            fmt = s0 + s2
        else:
            break
    return fmt, dic


## OTHER
#=======

