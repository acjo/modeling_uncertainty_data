# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.integrate import quad

# Problem 1
def initialize():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as spar on
    all engines. Return the DirectView.
    """
    #set up client
    client = Client()
    #group all clients as direct view
    dview = client[:]
    #import scipy.sparse on all clients
    dview.execute('import scipy.sparse as sparse')
    #also import numpy
    #returnt he direction view
    return dview

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    #initialize direct view
    dview = initialize()
    #set blocking
    dview.block = True
    #push up dictionary of values
    dview.push(dx)
    #get list of keys
    keys = list(dx.keys())
    #check that for each key the value is properly set
    for key in keys:
        #pull down the value that each engine has for key
        pulled = dview.pull(key)
        #get the value it should be for that given key
        value = dx[key]
        #create appropriate sized array
        current_array_values = [value for _ in pulled]
        #assertion call
        assert np.all(current_array_values == current_array_values)

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.

    Parameters:
        n (int): number of draws to make

    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    #intilize the direct view
    dview = initialize()
    #push up value for n
    dview.execute('import numpy as np')

    #function that returns the mean, min, and max
    def get_mean_min_max(x):
        return np.mean(x), np.min(x), np.max(x)
    #this function returnds a standard normal draw of size n
    f_norm = lambda n: np.random.normal(loc=0, scale=1.0, size=n)
    #tell each engine to draw a standard normal
    draw_response = dview.apply_async(f_norm, n)
    #now get the response using map_async
    response = dview.map_async(get_mean_min_max, draw_response.get())
    #now we have to strip out the means, mins, maxs
    values = response.get()
    means = [value[0] for value in values]
    mins = [value[1] for value in values]
    maxs = [value[2] for value in values]
    return means, mins, maxs

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    n_vals = [1000000,5000000,10000000,15000000]
    prob3_time = []
    serial_time = []
    #get number of cores being used
    N = len(Client().ids)
    for n in n_vals:
        #start timer for prob3()
        start = time.time()
        prob3(n=n)
        end = time.time ()
        #append the time
        prob3_time.append(np.abs(end - start))
        #star timer for serialized code
        start = time.time()
        mean = []
        min = []
        max = []
        for _ in range(N):
            #draw
            draw = np.random.normal(loc=0, scale=1.0, size=n)
            #calculate mean
            mean.append(np.mean(draw))
            #calculate min
            min.append(np.min(draw))
            #calculate mx
            max.append(np.max(draw))
        end = time.time()
        #append the time
        serial_time.append(np.abs(end - start))

    #plot the figure
    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.plot(n_vals, prob3_time, 'r-.', label='Parallel')
    ax.plot(n_vals, serial_time, 'b-.', label='Serial')
    ax.set_title('Parallel vs. Serial Time')
    ax.legend(loc='best')
    ax.set_xlabel('n (Number of Draws)')
    ax.set_ylabel('time (seconds)')
    plt.show()

# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    #get number of cores so we know how to split it up
    N = len(Client().ids)
    #initialize parallel processing
    dview = initialize()
    dview.execute('import numpy as np')
    #get our domain values
    x_vals = np.linspace(a, b, n)
    #now we partition our domain space
    x_partition = []
    for i in range(N):
        #minimum index for slicing
        min_index = int(n*i/N)
        #maximum index for slciing
        max_index = int(n*(i+1)/N)
        #current values we are partigioning
        curr_xvals = x_vals[min_index:max_index]
        #if i is not zero we need to add the right most
        #end point of the previous interval
        if i != 0:
            curr_xvals = np.concatenate(([x_partition[i-1][-1]],
                                        curr_xvals))
        #append values to our partition
        x_partition.append(curr_xvals)
    print(x_partition)
    #this defines the trapezoidal rule
    def trapezoid(x):
        h = x[-1] - x[-2]
        left = f(x[:-1])
        right = f(x[1:])
        return h*np.sum(left + right)/2

    #get the response using map i.e. each left_right_sum()
    #is applied to each subinterval
    response = dview.map_async(trapezoid, x_partition)
    return np.sum(response.get())

if __name__ == "__main__":
    f = lambda x: x**2
    a = 1
    b = 12
    n = 20
    print(parallel_trapezoidal_rule(f, a, b, n))
    print(quad(f, a, b))
