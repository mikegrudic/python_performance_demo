# Optimization and performance in Python

Python code has a reputation for being slower than other programming languages, and that is often well-deserved. But we will see here that there are ways to make your python code run orders-of-magnitude faster with relatively little extra work.


```python
import numpy as np
%load_ext line_profiler
```

For our problem let's assume we have a collection of point-masses, each with its own 3D coordinates, and we wish to know the gravitational acceleration exerted on each of the masses. This is requied e.g. for performing N-body simulations of planetary systems, star clusters, galaxies, and dark matter structure.

# Initialization


```python
num_particles = 10**3 # number of particles
masses = np.random.rand(num_particles) # particle masses chosen at random on [0,1)
coordinates = np.random.normal(size=(num_particles,3)) # particle positions chosen at random in a 3D Gaussian blob
```

Let's code a typical Numpy function that will compute the N-body gravitational acceleration. We will start as simple as possible while working within the numpy paradigm for arrays and docstrings, and explicitly writing out the loop operations.

## Naïve implementation


```python
def nbody_accel(masses, coordinates, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses.

    Parameters
    ----------
    masses: array_like
        Shape (N,) array of masses
    coordinates: array_like
        Shape (N,3) array of coordinates
    G: float, optional
        Value of Newton's gravitational constant (default to 1)

    Returns
    -------
    accel: ndarray
        Shape (N,3) array containing the gravitational acceleration experienced
        by each point-mass
    """

    # first declare the array that will store the acceleration
    accel = np.zeros_like(coordinates) # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0] # number of particles

    for i in range(N):
        for j in range(N):
            if i==j: continue # self-force is 0
            # first need to calculate the distance between i and j
            distance = 0.
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                distance += dx*dx
            if distance == 0: continue # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                accel[i,k] += G * masses[j] * dx / distance**3

    return accel
```

# Performance test

One way to test the performance of a function is to use the `%timeit` magic, which will run the function repeatedly and give you an average of how long it took.


```python
%timeit nbody_accel(masses,coordinates)
```

    7.46 s ± 83 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Oh shit that took forever didn't it? And that's just for 1000 masses - imagine if we had to simulate a Milky Way of ~10^11 masses! It would take (10^11/10^3)^2 = 10^16 times longer! Who's got time for that?

What's taking so long? One way to get a breakdown of which lines in your function are taking the most time is with the `%lprun` magic provided by the `line_profiler` package


```python
%lprun -f nbody_accel nbody_accel(masses, coordinates)
```

We see that there is no single line that takes the vast majority of the time, so optimizing this line-by-line could be time-consuming.

**Exercise**: How much can you optimize the above function just by rearranging the way the loop is structured and the floating-point calculations are carried out? (there should be a factor of ~2 on the table). 

But the reason for the low performance is more fundamental: explicit indexed loop operations like this have a huge amount of overhead when running natively in the Python interpreter. Much of python's flexibility comes with a trade-off for performance, so pure-python numerical code like this will almost always be outperformed by a compiled language like C++ or Fortran.

Is there a way to get around this?

# Numba

Numba performs JIT (just-in-time) compilation of your numerical Python code so that it is possible in theory to obtain performance equal to compiled languages. The simplest way to use it is just to put a `@jit` decordator on your function.


```python
from numba import jit

@jit
def nbody_accel_numba(masses, coordinates, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses.

    Parameters
    ----------
    masses: array_like
        Shape (N,) array of masses
    coordinates: array_like
        Shape (N,3) array of coordinates
    G: float, optional
        Value of Newton's gravitational constant (default to 1)

    Returns
    -------
    accel: ndarray
        Shape (N,3) array containing the gravitational acceleration experienced
        by each point-mass
    """

    # first declare the array that will store the acceleration
    accel = np.zeros_like(coordinates) # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0] # number of particles

    for i in range(N):
        for j in range(N):
            if i==j: continue # self-force is 0
            # first need to calculate the distance between i and j
            distance = 0.
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                distance += dx*dx
            if distance == 0: continue # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                accel[i,k] += G * masses[j] * dx / distance**3

    return accel

%timeit nbody_accel_numba(masses,coordinates)
```

    /var/folders/h1/4vzhdl0j21q2tk7dy4cyzbxc0000gn/T/ipykernel_95055/309808660.py:3: NumbaDeprecationWarning: [1mThe 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.[0m
      @jit


    9.04 ms ± 114 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


Note the factor of ~1000 speedup, obtained by fundamentally changing the way the code gets transformed into instructions! 

We can get a little more performance by adding the `fastmath=True` argument, which relaxes the requirement that floating-point operations be performed according to the standard spec. This is usually fine, but **always test your function's accuracy to make sure it calculates the result to desired accuracy**


```python
@jit(fastmath=True)
def nbody_accel_numba_fastmath(masses, coordinates, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses.

    Parameters
    ----------
    masses: array_like
        Shape (N,) array of masses
    coordinates: array_like
        Shape (N,3) array of coordinates
    G: float, optional
        Value of Newton's gravitational constant (default to 1)

    Returns
    -------
    accel: ndarray
        Shape (N,3) array containing the gravitational acceleration experienced
        by each point-mass
    """

    # first declare the array that will store the acceleration
    accel = np.zeros_like(coordinates) # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0] # number of particles

    for i in range(N):
        for j in range(N):
            if i==j: continue # self-force is 0
            # first need to calculate the distance between i and j
            distance = 0.
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                distance += dx*dx
            if distance == 0: continue # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                accel[i,k] += G * masses[j] * dx / distance**3

    return accel

%timeit nbody_accel_numba_fastmath(masses, coordinates)
```

    /var/folders/h1/4vzhdl0j21q2tk7dy4cyzbxc0000gn/T/ipykernel_95055/1027182532.py:1: NumbaDeprecationWarning: [1mThe 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.[0m
      @jit(fastmath=True)


    6.91 ms ± 85.1 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


# Parallelism
There is even more performance on the table here: your computer has **multiple** processors, and so far we have just been using only one. The easiest way to parallelize this is to run the outer loop in parallel using `prange`, and tell numba to use it with `parallel=True`


```python
from numba import prange

@jit(fastmath=True,parallel=True)
def nbody_accel_numba_fastmath_parallel(masses, coordinates, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses.

    Parameters
    ----------
    masses: array_like
        Shape (N,) array of masses
    coordinates: array_like
        Shape (N,3) array of coordinates
    G: float, optional
        Value of Newton's gravitational constant (default to 1)

    Returns
    -------
    accel: ndarray
        Shape (N,3) array containing the gravitational acceleration experienced
        by each point-mass
    """

    # first declare the array that will store the acceleration
    accel = np.zeros_like(coordinates) # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0] # number of particles

    for i in prange(N):
        for j in range(N):
            if i==j: continue # self-force is 0
            # first need to calculate the distance between i and j
            distance = 0.
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                distance += dx*dx
            if distance == 0: continue # just 
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                accel[i,k] += G * masses[j] * dx / distance**3

    return accel

%timeit nbody_accel_numba_fastmath_parallel(masses, coordinates, G=1.)
```

    /var/folders/h1/4vzhdl0j21q2tk7dy4cyzbxc0000gn/T/ipykernel_95055/521207335.py:3: NumbaDeprecationWarning: [1mThe 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.[0m
      @jit(fastmath=True,parallel=True)


    1.41 ms ± 221 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


# Other options for optimization

There are many other ways to write performant python code than just the numba parallel CPU coding we have done here. Any python code can in principle be parallelized using python's native [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). [joblib](https://joblib.readthedocs.io/en/stable/) offers functionality for parallelism with a different implementation.  [numba also supports running code on the GPU](https://numba.readthedocs.io/en/stable/cuda/index.html). [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) is best known for its use in machine learning but is also more broadly useful for coding GPU-portable python code consisting of function compositions and array operations. [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) implements MPI (Message Passing Interface) for programs that may need to run distributed across multiple computers (i.e. supercomputers). The best choice will depend on your particular problem and requirements.

Python may still have a reputation for being slow, but the reality is that in many cases the time-to-solution (coding+computation) for a give project can be shorter, just by taking advantage of Python's extensive community library support while optimizing the most numerically-intensive parts as we have here.
