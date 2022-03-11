

from deepmd.nvnmd.utils.fio import Fio, FioDic, FioTxt, FioHead
from deepmd.nvnmd.utils.format_string import FormatString

class Code:

    def __init__(self,
        dictionary: str,
        template: str,
        output: str
        ) -> None:
        self.dictionary = dictionary
        self.template = template
        self.output = output
        
    
    def run(self):
        self.dic = FioDic().load(self.dictionary, {})
        FioDic().save('nvnmd/dictionary.json', self.dic)
        self.dic = self.flatten_dic(self.dic)
        FioDic().save('nvnmd/dictionary_flatten.json', self.dic)

        f = Fio()
        if f.is_file(self.template) and f.is_file(self.output):
            self.replace(self.template, self.output, self.dic)
        if f.is_path(self.template) and (not f.is_file(self.output)):
            fnlist = f.get_file_list(self.template)
            n = len(self.template)
            for fn in fnlist:
                fn2 = self.output + fn[n:]
                if fn.split('.')[-1] in 'txt,v,sv,md'.split(','):
                    self.replace_auto(fn, fn2, self.dic)
    
    def flatten_dic(self, dic):
        dic2 = {}
        for key in dic.keys():
            if type(dic[key]) == dict:
                dic2.update(self.flatten_dic(dic[key]))
            else:
                dic2[key] = dic[key]
        return dic2 
    
    def replace(self, fn_s, fn_d, dic):
        lines = FioTxt().load(fn_s, [])
        lines2 = FormatString().replace(dic, lines)
        FioTxt().save(fn_d, lines2)
    
    def replace_auto(self, fn_s, fn_d, dic):
        lines = FioTxt().load(fn_s, [])
        if (lines[0].startswith('//CODE')):
            lines2 = FormatString().replace(dic, lines)
        else:
            lines2 = lines 
        FioTxt().save(fn_d, lines2)

def code(*, 
        dictionary: str = 'nvnmd/config.npy', 
        template: str= 'code_tmp/params.v', 
        output: str = 'src/params.v', 
        **kwargs
        ):
    codeObj = Code(dictionary, template, output)
    codeObj.run()