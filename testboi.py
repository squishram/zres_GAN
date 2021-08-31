import numpy as np

def chunks_susan(data, size, chunknum, sigmaxy, sigmaz):
    	#chunknum should be a 3 element vector containing x, y, z chunking values respectively
        #chunk data into bits
        #output chunknum arrays each with the data that could contribute to chunk chunknum
        #to start with let's assume we want to chunk by same factor in x,y, and z directions - correct later
    	#contribution distance cuttoff depends on sigma so these need to be input too
        i = 0
        chunkedxstart = []
        chunkedystart = []
        chunkedzstart = []
        #go through all chunks
        for x in range(chunknum):
            for y in range(chunknum):
                for z in range(chunknum):
                    # I never actually use i do I want it?
                    i = i + 1
                    chunkedxstart.append((size[0]*x)//chunknum)
                    chunkedystart.append((size[1]*y)//chunknum)
                    chunkedzstart.append((size[2]*z)//chunknum)
        #get all pixels corresponding to chunk + 4 sigma
        #BEWARE EDGE EFFECTS
    	#this assumes all chunk dimenstions same size
        chunksize = size[0] // chunknum
        #find which data points are inside that chunk
        # for j in range(len(data[0])):
        #     #print("I got into the for loop")
        #     #print(data[0][j])
        # print("this is a check onthe number of points")
        # print(p)
        # chunkeddata = 3
        # return [chunkedxstart,chunkedystart,chunkedzstart,chunksize,chunkeddata]
    return None

def zero_fill(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

def chunks_ish(data, n_chunks=2, boxsize=3):
    
    sz = np.shape(data)
    
    boxend = int(boxsize - np.ceil(boxsize/2))
    
    ### X AXIS ###
    # first, sort by x axis:
    data = data[:, data[0].argsort()]
    print(data)
    # we then need to divide data along rows into n_chunks, while retaining some 'overlap'
    # these chunks will go into a new numpy array dimension
          
    # sz_chunks = sz[1]/n_chunks + boxend
    
   
    data = np.split(data, np.where(data[0, :] < (1/n_chunks)*data[0].max())[0][1:])
    



    
    return data

    
    
    
data = np.random.randint(0 , 100, 30).reshape(3, 10)
print(data)
data = chunks_ish(data, 2)
print(data)




