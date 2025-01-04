from .MAG_BERT.manager import MAG_BERT
from .MISA.manager import MISA
from .MULT.manager import MULT
from .SDIF.manager import SDIF
from .MSZ.manager import MSZ


method_map = {
    
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'sdif': SDIF,
    'msz': MSZ

}