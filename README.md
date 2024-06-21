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

    5.14 s ± 180 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Oh shit that took forever didn't it? And that's just for 1000 masses - imagine if we had to simulate a Milky Way of ~10^11 masses! It would take (10^11/10^3)^2 = 10^16 times longer! Who's got time for that?

What's taking so long? One way to get a breakdown of which lines in your function are taking the most time is with the `%lprun` magic provided by the `line_profiler` package


```python
%lprun -f nbody_accel nbody_accel(masses, coordinates)
```

```
Timer unit: 1e-06 s

Total time: 24.1356 s
File: <ipython-input-3-1a5f7a885e4f>
Function: nbody_accel at line 1

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     1                                           def nbody_accel(masses, coordinates, G=1.):
     2                                               """
     3                                               Computes the Newtontian gravitational acceleration exerted on each of a set
     4                                               of point masses.
     5                                           
     6                                               Parameters
     7                                               ----------
     8                                               masses: array_like
     9                                                   Shape (N,) array of masses
    10                                               coordinates: array_like
    11                                                   Shape (N,3) array of coordinates
    12                                               G: float, optional
    13                                                   Value of Newton's gravitational constant (default to 1)
    14                                           
    15                                               Returns
    16                                               -------
    17                                               accel: ndarray
    18                                                   Shape (N,3) array containing the gravitational acceleration experienced
    19                                                   by each point-mass
    20                                               """
    21                                           
    22                                               # first declare the array that will store the acceleration
    23         1        183.0    183.0      0.0      accel = np.zeros_like(coordinates) # array of zeros shaped like coordinate (N,3)
    24         1          3.0      3.0      0.0      N = coordinates.shape[0] # number of particles
    25                                           
    26      1001        721.0      0.7      0.0      for i in range(N):
    27   1001000     533501.0      0.5      2.2          for j in range(N):
    28   1000000     555480.0      0.6      2.3              if i==j: continue # self-force is 0
    29                                                       # first need to calculate the distance between i and j
    30    999000     518899.0      0.5      2.1              distance = 0.
    31   3996000    2554522.0      0.6     10.6              for k in range(3):
    32   2997000    3466373.0      1.2     14.4                  dx = coordinates[j,k] - coordinates[i,k]
    33   2997000    2305378.0      0.8      9.6                  distance += dx*dx
    34    999000     659161.0      0.7      2.7              if distance == 0: continue # just skip if points lie on top of each other
    35    999000    2183010.0      2.2      9.0              distance = np.sqrt(distance)
    36                                           
    37                                                       # now compute the acceleration
    38   3996000    2678010.0      0.7     11.1              for k in range(3):
    39   2997000    3421611.0      1.1     14.2                  dx = coordinates[j,k] - coordinates[i,k]
    40   2997000    5258707.0      1.8     21.8                  accel[i,k] += G * masses[j] * dx / distance**3
    41                                           
    42         1          1.0      1.0      0.0      return accel
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



    8.82 ms ± 520 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


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



    7.13 ms ± 441 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


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




    1.11 ms ± 218 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


# Other options for optimization

There are many other ways to write performant python code than just the numba parallel CPU coding we have done here. We have only looked at two common tricks for speeding up numba code, but there are [many others](https://numba.readthedocs.io/en/stable/user/performance-tips.html). Any python code can in principle be parallelized using python's native [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). [joblib](https://joblib.readthedocs.io/en/stable/) offers functionality for parallelism with a different implementation.  [numba also supports running code on the GPU](https://numba.readthedocs.io/en/stable/cuda/index.html). [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) is best known for its use in machine learning but is also more broadly useful for coding GPU-portable python code consisting of function compositions and array operations. [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) implements MPI (Message Passing Interface) for programs that may need to run distributed across multiple computers (i.e. supercomputers). The best choice will depend on your particular problem and requirements.

Python may still have a reputation for being slow, but the reality is that in many cases the time-to-solution (coding+computation) for a give project can be shorter, just by taking advantage of Python's extensive community library support while optimizing the most numerically-intensive parts as we have here.
