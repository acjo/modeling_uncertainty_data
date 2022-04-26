# Problem 3
"""In each process, generate a random number, then send that number to the
process with the next highest rank (the last process should send to the root).
Print what each process starts with and what each process receives.

Usage:
    $ mpiexec -n 2 python problem3.py
    Process 1 started with [ 0.79711384]        # Values and order will vary.
    Process 1 received [ 0.54029085]
    Process 0 started with [ 0.54029085]
    Process 0 received [ 0.79711384]

    $ mpiexec -n 3 python problem3.py
    Process 2 started with [ 0.99893055]
    Process 0 started with [ 0.6304739]
    Process 1 started with [ 0.28834079]
    Process 1 received [ 0.6304739]
    Process 2 received [ 0.28834079]
    Process 0 received [ 0.99893055]
"""
from mpi4py import MPI
import numpy as np

#set communicator and rank object and get the size
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
#initialize variables
r = np.zeros(1)
s = np.random.rand(1)

#iterate through all possible ranks
for j in range(SIZE):
    #if the current rank is j then we send and receive to j-1 and j+1
    if RANK == j:
        #if the rank is 0 send to 1 and receive from SIZE-1
        if RANK == 0:
            print('Process {0}: Sending {1} to {2}'. format(RANK, s, RANK+1))
            COMM.Send(s, dest=1)
            COMM.Recv(r, source = SIZE-1)
            print('Process {0}: Received {1} from {2}'. format(RANK, r, SIZE-1))
        #if the rank is SIZE-1 receive from SIZE-2 and send to 0
        if RANK == SIZE-1:
            print('Process {0}: Sending {1} to {2}'. format(RANK, s, 0))
            COMM.Send(s, dest=0)
            COMM.Recv(r, source=SIZE-2)
            print('Process {0}: Received {1} from {2}'. format(RANK, r, RANK-1))
        #otherwise send to RANK+1 and receive from RANK-1
        else:
            print('Process {0}: Sending {1} to {2}'. format(RANK, s, RANK+1))
            COMM.Send(s, dest=RANK+1)
            COMM.Recv(r, source=RANK-1)
            print('Process {0}: Received {1} from {2}'. format(RANK, r, RANK-1))




