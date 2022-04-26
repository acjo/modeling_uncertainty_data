# Problem 2
"""Pass a random NumPy array of shape (n,) from the root process to process 1,
where n is a command-line argument. Print the array and process number from
each process.

Usage:
    # This script must be run with 2 processes.
    $ mpiexec -n 2 python problem2.py 4
    Process 1: Before checking mailbox: vec=[ 0.  0.  0.  0.]
    Process 0: Sent: vec=[ 0.03162613  0.38340242  0.27480538  0.56390755]
    Process 1: Recieved: vec=[ 0.03162613  0.38340242  0.27480538  0.56390755]
"""
from mpi4py import MPI
from sys import argv
import numpy as np
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

#get n from keyword argument
n = int(argv[1])
#initialize a
a = np.zeros(n)

if RANK == 0:
    #copy for sending
    send = a.copy()
    #change all to random numbers
    send[:] = np.random.rand()
    print("Process 0: Sending: {} to process 1.\n".format(send))
    #send
    COMM.Send(send, dest=1)
    print("Process 0: Message sent.\n")
elif RANK == 1:
    print("Process 1: Waiting for the message... current buffer={}.\n".format(a))
    #receive
    COMM.Recv(a, source=0)
    print("Process 1: message received!\nReceived: {}.".format(a))
