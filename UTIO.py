import numpy as np

def ReadUltraVisionData(flname, datatype=np.float):


    """
        Reads text file output from the Tomoview software, path specified
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


    return Data

def ReadTomoviewData(flname, datatype=np.float):

    fl = open(flname,'r')

    D = fl.readlines()

    fl.close()

    H = 16

    L = len(D)

    # N = int(D[5])

    # N = int(D[5].split('\t')[-1].split('\n')[0])

    NScan = int(np.float(D[4].split('=')[1]))

    B = int(np.round(L/(H+NScan)))


    Nt = min([int(np.float(D[10 + b*(H+NScan)].split('=')[1])) for b in range(B)])



    if NScan>1:

        Data = np.array([np.array([np.fromstring(D[i], sep='\t', dtype=datatype)[0:Nt] for i in range(int(H+b*(H+NScan)),int(int(H+b*(H+NScan))+NScan))]) for b in range(B)])

        Data = np.swapaxes(np.array(Data),1,2)

    else:

        Data = [[list(np.fromstring(D[i], sep='\t', dtype=datatype))[0:Nt] for i in range(int(H+b*(H+NScan)),int(int(H+b*(H+NScan))+NScan))][0] for b in range(B)]

        Data = np.array(Data)
    #     # prin
        #
        # Data = np.array(Data)
        # Data = np.zeros((B,Nt))
        #
        # for i in range(B):
        #
        #     Data[i,:] = D[i][0]





    # Data = []
    #
    # for b in range(B):
    #
    #
    #     istart = int(H+b*(H+NScan))
    #     iend = int(istart+NScan)
    #
    #
    #     # DD = D[istart:iend]
    #
    #     Data.append([list(np.fromstring(D[i], sep='\t', dtype=datatype)) for i in range(istart,iend)])

    #
    #
    #
    # Data = np.swapaxes(np.array(Data),1,2)


    return Data
