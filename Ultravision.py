import numpy as np

def ReadScan(flname, datatype=np.float):


    """
        Reads text file output from the Ultravision software, path specified
        by flname. Format of flname is header for beam 1 (18 lines), N A-scans for beam 1,
        header for beam 2 (18 lines), N A-Scans for beam  2, ...  Returns
        numpy array of shape (Number of Beams , Number of AScan Points, Number of Scans)

    """


    # fl = open(flname,'rb')

    fl = open(flname,'r')


    D = fl.readlines()

    fl.close()

    H = 19

    L = len(D)

    # N = int(D[5])

    N = int(D[5].split('\t')[-1].split('\n')[0])

    B = int(L/(H+N))

    Data = []

    for b in range(B):

        istart = int(H+b*(H+N))
        iend = int(istart+N)


        DD = D[istart:iend]

        Data.append([list(np.fromstring(DDD, sep=' ', dtype=datatype)) for DDD in DD])



    Data = np.swapaxes(np.array(Data),1,2)


    # if Data.shape[0]==0:
    #
    #     Data = Data[0,:,:]
    #


    return Data
