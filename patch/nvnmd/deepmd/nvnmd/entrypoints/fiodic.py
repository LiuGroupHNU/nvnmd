

from deepmd.nvnmd.utils.fio import FioDic 

def fiodic(
    *,
    load_file: str,
    update_file: str,
    save_file: str, 
    **kwargs
    ):
    f = FioDic()
    jdata = f.load(load_file, {})
    jdata2 = f.load(update_file, {})
    jdata = f.update(jdata2, jdata)
    f.save(save_file, jdata)


        