
import os
import sys
from fs import read_fn, replace


# write_code
## Usage
"""
从"code-tmp"文件夹读取模板文件，并重新生成verilog代码文件
变量名设定：若变量为整型常量，变量名为全部大写，若变量名为浮点数，使用小写
"""

class WCode():

    def __init__(self, tmp_path, out_path):
        # user_path = 'D:/WorkSpace/deepmd-kit/smooth/ws-gete/vivado/code-tmp'
        user_path = '/home/mph/WorkSpace/deepmd-kit/smooth/ws-gete/vivado/code-tmp'
        tmps = "device,general,math,md,tb".split(',')
        self.tmp_paths = [user_path+'/'+v for v in tmps]
        self.tmp_paths.append(tmp_path)
        self.out_path = out_path
    
    def display(self):
        print("TMP_PATHS:")
        for v in self.tmp_paths:
            print(v)
        print("OUT_PATH:")
        print(self.out_path)
    
    def instantiate(self, device_name, dic, name=None):
        """:read the template file and generate the special verilog module file
            using the configuration {dic}
        @device_name: the module and file will be rename using the {device_name}
        @dic: contains the configuration for special instance
        @tmp_path: where have the tempelate file
        @out_path: where the instantiate module is generated in
        """
        if 'INSTANT' in dic.keys():
            if len(dic['INSTANT']) > 0:
                if device_name not in dic['INSTANT']:
                    print("%20s :        no in INSTANT"%(device_name))
                    return

        if 'NO_INSTANT' in dic.keys():
            if len(dic['NO_INSTANT']) > 0:
                if device_name in dic['NO_INSTANT']: 
                    print("%20s :        in NO_INSTANT"%(device_name))
                    return
        
        is_tmp_exist = False
        for tmp_path in self.tmp_paths:
            tmp_fn = "%s/%s.v"%(tmp_path, device_name)
            if os.path.exists(tmp_fn):
                is_tmp_exist = True
                break
        
        if not is_tmp_exist:
            print("Don't find %s.v file for tmp_paths"%(device_name))
            self.display()
        
        else:
            out_name = dic['name'] if (name == None) else name
            out_fn = "%s/%s.v"%(self.out_path, out_name)
            print("%20s : %20s"%(device_name, out_name))
            lines = read_fn(tmp_fn)
            lines2 = replace(dic, lines)
            fw = open(out_fn, 'w', encoding='utf-8')
            fw.writelines(lines2)
            fw.close()

# -- mian --
import sys
if (__name__ == "__main__"):
    argvs = sys.argv[1:]
    if len(argvs) < 2:
        print("Please input 2 parameters")
        sys.exit()
    
    tmp_path, out_path = argvs[0], argvs[1]

    tmp_path = tmp_path.split('/')
    device_name = tmp_path[-1].replace('.v','')
    tmp_path = '/'.join(tmp_path[:-1])

    out_path = out_path.split('/')
    name = out_path[-1].replace('.v','')
    out_path = '/'.join(out_path[:-1])

    wc = WCode(tmp_path, out_path)
    wc.instantiate(device_name, {'name':name})