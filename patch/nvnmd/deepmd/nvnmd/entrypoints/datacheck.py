

import numpy as np 
from deepmd.nvnmd.utils.fio import Fio, FioHead


class DataCheck():
    def __init__(self, datapath, rcut) -> None:
        self.datapath = datapath
        self.rcut = rcut 
        self.plist = []
        fio = Fio()
        plist = fio.get_file_list(datapath)
        for p in plist:
            if p.endswith('coord.npy'):
                self.plist.append(p.replace('coord.npy', ''))

    def check_density(self):
        head = FioHead().info()
        print(f"{head}: check density")
        for p in self.plist:
            coords = np.load(p+'/coord.npy')
            boxs = np.load(p+'/box.npy')
            nfna3 = np.size(coords)
            nf33 = np.size(boxs)
            nf = nf33 // 9 
            na = nfna3 // nf // 3
            boxs = np.reshape(boxs, [nf, 3, 3])
            vmin = 1e6 
            for b in boxs:
                v = np.inner(np.cross(b[0], b[1]), b[2])
                if (v < vmin): vmin = v 
            
            dmax = na / vmin 
            vo = 4 / 3 * np.pi * self.rcut ** 3

            print('='*32)
            print(p)
            print("nframe:", nf, 'natom:', na)
            print('vmin: ', vmin, 'dmax:', dmax, 'natom (%d Ang)'%self.rcut, dmax * vo)
            print('='*32)
            print() 


def datacheck(*, 
        datapath: str = './', 
        rcut: str = '6.0',
        **kwargs
        ):
    datacheckObj = DataCheck(datapath, float(rcut))
    datacheckObj.check_density()

