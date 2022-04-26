# Problem 1
"""Print 'Hello from process n' from processes with an even rank and
print 'Goodbye from process n' from processes with an odd rank (where
n is the rank).

Usage:
    $ mpiexec -n 4 python problem1.py
    Goodbye from process 3                  # Order of outputs will vary.
    Hello from process 0
    Goodbye from process 1
    Hello from process 2

    # python problem1.py
    Hello from process 0
"""
#import mpi
from mpi4py import MPI

#get the rank
RANK = MPI.COMM_WORLD.Get_rank()

#check if the rank is even or odd and print the corresponding message.
if RANK % 2 == 0:
    print("Hello from from process {}".format(RANK))
else:
    print("Goodbye from process {}".format(RANK))
