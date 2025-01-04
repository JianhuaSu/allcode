from .MAG_BERT import MAG_BERT
from .MISA import MISA
from .MULT import MULT
from .SDIF import SDIF
from .MSZ import MMEncoder

method = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'sdif': SDIF,
    'msz': MMEncoder,
}