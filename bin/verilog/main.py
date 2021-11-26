

import sys
import numpy as np
from write_code import main as write_code_main
from descrpt import main as descrpt_main
from MdDebug import check, test_force

def help():
    print("INFO: help")
    print("freeze IN(weight_file) OUT(config_npy_file)")
    print("map IN(config_file) OUT(map_file)")
    print("wrap IN(config_file, map_file) OUT(fpga_model.txt,[fpga_model.npy])")
    print("check IN(npy_file, [key1[min|max|len|fla], key2, ...])")
    print("test_force IN(atom.xsf, idx.txt, fi.txt, tag.txt, frc_hex.txt, lst_hex.txt)")
    print("code IN(config_file, verilog_tmp_path, verilog_out_path [module1 module2 ...])")
    print("     You also need decode.npy in the ${bin}/data folder")
    print("ext_atm IN(type.raw, type_map.raw, set_path, idx)")

if __name__ == "__main__":
    argvs = sys.argv[1:]
    if (len(argvs) == 0) or argvs[0] == '-h' or argvs[0] == '--help':
        help()
    
    else:
        mod = argvs[0]
        if mod == 'freeze':
            write_code_main(argvs)

        if mod == "map":
            descrpt_main(argvs)
        
        if mod == "wrap":
            write_code_main(argvs)
        
        if mod == "check":
            check(argvs[1:])

        if mod == "test_force":
            test_force(argvs[1:])
        
        if mod == "code":
            write_code_main(argvs)

        if mod == "ext_atm":
            write_code_main(argvs)
