from numpy import *
from numpy.fft import *
from numpy.linalg import *
# from numpy import savetxt,loadtxt
# import os.path
# import sys
from matplotlib.pyplot import *

from imp import reload
import _pickle as pickle


try:

    ndtcmap = pickle.load(open('/Users/jlesage/Dropbox/python/ndtcolourmap.p','rb'))

except:

    ndtcmap = pickle.load(open('/mnt/c/Users/jlesage/Dropbox/python/ndtcolourmap.p','rb'))

# import readline
# import rlcompleter
# import atexit
# import os
