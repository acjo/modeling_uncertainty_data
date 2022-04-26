# Problem 4
"""The n-dimensional open unit ball is the set U_n = {x in R^n : ||x|| < 1}.
Estimate the volume of U_n by making N draws on each available process except
for the root process. Have the root process print the volume estimate.

Command line arguments:
    n (int): the dimension of the unit ball.
    N (int): the number of random draws to make on each process but the root.

Usage:
    # Estimate the volume of U_2 (the unit circle) with 2000 draws per process.
    $ mpiexec -n 4 python problem4.py 2 2000
    Volume of 2-D unit ball: 3.13266666667      # Results will vary slightly.
"""

from mpi4py import MPI
from sys import argv
import numpy as np

#set communicator and rank object
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

#accept integers from command line
n, N = int(argv[1]), int(argv[2])

#if the rank is not equal to zero, perform our estimate
if RANK != 0:
    #draw uniformly on [-1, 1]^n, N times
    draw = np.random.uniform(-1, 1, size=(n, N))
    #volume estimate
    volume = 2**n
    #get the lengths
    lengths = np.linalg.norm(draw, axis=0)
    #get the number of points within the unit hyper-sphere
    num_within = np.count_nonzero(lengths < 1)
    estimate = np.zeros(1)
    estimate[0] =  volume*num_within / N
    #send volume estimate to root
    COMM.Send(estimate, dest=0)

#if the RANK is the root process
if RANK == 0:
    estimates = np.zeros(SIZE-1)
    #receive the volume estimate for each other process
    for i in range(0, SIZE-1):
        a = np.zeros(1)
        COMM.Recv(a, source=i+1)
        estimates[i] = a[0]

    #average all the estimates and print
    final_estimate = np.sum(estimates)/estimates.size

    print(final_estimate)

