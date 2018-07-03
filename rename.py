import os
import sys


Dir = sys.argv[1]
pth = os.listdir(Dir)

for p in pth:

    os.rename(Dir+'/'+p.split('-')[-1])
