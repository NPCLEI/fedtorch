import sys
sys.path.append('./')

from experiments.EMNISTAE.libs.dataset import get_source
from experiments.EMNISTAE.libs.nn import AEL,AutoEncoderInstructor

AutoEncoderInstructor(AEL(),get_source(),'cpu',opt='adam',batch_size=1024,lr=1e-3).fit()