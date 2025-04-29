# How 2 go fast with numba

## Python and performance

- Python code has a reputation for being slower than other programming languages, which is often well-deserved. 

- Python is an **interpreted** programming language: your code is converted to bytecode, which is in turn executed by a set of machine instructions by the Python interpreter when you run the program. 

- This is in contrast to a **compiled** programming language such as C++ or Fortran, where your code is converted into machine instructions that are compiled into a binary, which you then execute later. 

- This implementation gives Python a great degree of power and flexibility, but comes with a trade-off for performance.

- But we will see here that there are ways to make your python code run **orders-of-magnitude** faster with relatively little extra work.

## Problem Setup

For our problem let's assume we have a collection of $N$ point-masses $m_i$, each with its own 3D coordinates $\mathbf{x}_{i}$, and we wish to know the gravitational acceleration $\mathbf{g}_{i}$ experienced each of the masses. This is requied e.g. for performing N-body simulations of planetary systems, star clusters, galaxies, and dark matter structure. 

At the position $\mathbf{x}_{i}$ of each particle **i**, we wish to compute

$$
\mathbf{g}_{i} = \sum_{i \neq j} G m_j \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|^3}
$$

### Initialization
For the sake of the example we will initialize $m_i$ and $\mathbf{x}_{i}$ random samples, from a uniform distribution on $[0,1)$ and from a 3D Gaussian respectively:


```python
import numpy as np

num_particles = 10**3 # number of particles
np.random.seed(42) # seed the RNG to run reproducibly
masses = np.random.rand(num_particles) # particle masses chosen at random on [0,1)
coordinates = np.random.normal(size=(num_particles,3)) # particle positions chosen at random in a 3D Gaussian blob
```

## Naïve native-python implementation

Let's code a typical Numpy function that will compute the N-body gravitational acceleration. We will start as simple as possible while working within the numpy paradigm for arrays and docstrings, and explicitly writing out the loop operations.


```python
def nbody_accel(masses, coordinates, G=1.0):
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
    accel = np.zeros_like(coordinates)  # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0]  # number of particles

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # self-force is 0
            
            # first need to calculate the distance between i and j
            distance = 0.0
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                distance += dx * dx
            if distance == 0:
                continue  # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                accel[i, k] += G * masses[j] * dx / distance**3 # computes the actual summand

    return accel
```

# Performance test

One way to test the performance of a function is to use the `%timeit` magic, which will run the function repeatedly and give you an average of how long it took.


```python
%timeit nbody_accel(masses,coordinates)
```

    4.17 s ± 12.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


That took forever.

What's taking so long? One way to get a breakdown of which lines in your function are taking the most time is with the `%lprun` magic provided by the `line_profiler` package


```python
%load_ext line_profiler

%lprun -f nbody_accel nbody_accel(masses, coordinates)
```

We see that there is no single line that takes the vast majority of the time, so optimizing this line-by-line is not practical.

## Exercise
When facing a performance bottleneck, the most elegant solution is to choose a more optimal algorithm, or implementation of a given algorithm. How much can you optimize the above function just by rearranging the way the loop is structured and the floating-point calculations are carried out?

## The need for JIT compilation

Even with a per reason for the low performance is more fundamental: explicit indexed loop operations like this have a huge amount of overhead when running natively in the Python interpreter. Much of python's flexibility comes with a trade-off for performance, so pure-python numerical code like this will almost always be outperformed by a compiled language like C++ or Fortran.

Is there a way to get around this?

## Numba

Numba performs JIT (just-in-time) compilation of your numerical Python code so that it is possible in theory to obtain performance comparable to compiled languages. The simplest way to use it is just to put a `@jit` decordator on your function.


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
    accel = np.zeros_like(coordinates)  # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0]  # number of particles

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # self-force is 0
            
            # first need to calculate the distance between i and j
            distance = 0.0
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                distance += dx * dx
            if distance == 0:
                continue  # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                accel[i, k] += G * masses[j] * dx / distance**3 # computes the actual summand

    return accel
%timeit nbody_accel_numba(masses,coordinates)
```

    6.73 ms ± 83.8 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)


Note the factor of ~1000 speedup, obtained by fundamentally changing the way the code gets transformed into instructions! 

You can also pass your existing function to the `jit()` function and it will return the jit'd version:


```python
nbody_accel_numba = jit(nbody_accel)
%timeit nbody_accel_numba(masses,coordinates)
```

    6.76 ms ± 28.5 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)


### Fancy optimizations: fast math

We can get a little more performance by adding the `fastmath=True` argument, which relaxes the requirement that floating-point operations be performed according to the standard IEEE 754 spec, and can substitute certain functions with faster versions that give the same result within machine precision. One example would be replacing a `x**3.` call with the much-faster `x*x*x`: this is not strictly conformant to what you asked the code to do, but agrees within machine precision. This is usually fine, but **always test your function's accuracy to make sure it calculates the result to desired accuracy**


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
            if i==j: 
                continue # self-force is 0
            # first need to calculate the distance between i and j
            distance = 0.
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                distance += dx*dx
            if distance == 0: 
                continue # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                accel[i,k] += G * masses[j] * dx / distance**3

    return accel
```


```python
%timeit nbody_accel_numba_fastmath(masses, coordinates)
```

    5.34 ms ± 23.1 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)


This provided a modest speedup. We can make sure we are getting the same result:


```python
np.allclose(nbody_accel_numba_fastmath(masses, coordinates), nbody_accel_numba(masses,coordinates))
```




    True



## Thread Parallelism
There is even more performance on the table here: your computer has **multiple** processors, and so far we have just been using only one. numba supports multi-threaded parallelism: to check or set the number of threads numba will run on, you can use `get_num_threads()` and `set_num_threads()`:


```python
from numba import get_num_threads, set_num_threads
num_threads = get_num_threads()
print(f"numba can use {num_threads} threads!")
set_num_threads(num_threads // 2)
print(f"now numba can use {get_num_threads()} threads!")
set_num_threads(num_threads)
print(f"now numba can use {num_threads} threads again!")
```

    numba can use 32 threads!
    now numba can use 16 threads!
    now numba can use 32 threads again!


The easiest way to parallelize the N-body force problem is to run the outer loop in parallel using `prange`, and tell numba to use it with `parallel=True`. Note that `prange` will behave exactly the same way as `range` if `parallel=False`.


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
 
    for i in prange(N): # this is prange now
        for j in range(N):
            if i==j: 
                continue # self-force is 0
            # first need to calculate the distance between i and j
            distance = 0.
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                distance += dx*dx
            if distance == 0: 
                continue # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j,k] - coordinates[i,k]
                accel[i,k] += G * masses[j] * dx / distance**3

    return accel
```


```python
%timeit nbody_accel_numba_fastmath_parallel(masses, coordinates, G=1.)
```

    832 μs ± 158 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
np.allclose(nbody_accel_numba_fastmath_parallel(masses, coordinates), nbody_accel(masses,coordinates))
```




    True



### Caveat parallelizor
The usual caveats of parallel programming apply here; always validate your parallel code. Although [numba can sometimes recognize simple reduction loops](https://numba.readthedocs.io/en/stable/user/parallel.html?highlight=race#explicit-parallel-loops) and use atomics to avoid potential data races, in general potential data races must be recognized and avoided. AFAIK explicit atomic operations are not supported as in OpenMP, so not all of your favorite thread-parallel algorithms are possible.

## Limitations of numba

numba is designed for array and compute-intensive code, and can only JIT-compile a specific subset of python. As a rule of thumb: **if your python code basically looks like a simple C or fortran array loop, you can probably JIT it with numba**.

To see this, let's try to use `scipy.KDTree` inside a numba function. Suppose you only care about the force from the $N_{\rm ngb}$ nearest neighbors, and you want to use `KDTree.query` to search those nearest neighbors efficiently. The following code will not compile:


```python
from scipy.spatial import KDTree

@jit
def nbody_accel_nearest(masses, coordinates, num_neighbors, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses, accounting only for the num_neighbors set of nearest neighbors.

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

    tree = KDTree(coordinates) # build the tree

    # first declare the array that will store the acceleration
    accel = np.zeros_like(coordinates)  # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0]  # number of particles

    for i in range(N):
        _, ngb = tree.query(coordinates[i],num_neighbors+1)
        for j in ngb[1:]:        
            # first need to calculate the distance between i and j
            distance = 0.0
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                distance += dx * dx
            if distance == 0:
                continue  # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                accel[i, k] += G * masses[j] * dx / distance**3 # computes the actual summand

    return accel
```

### Nesting python object-mode code in JIT'd code

Here we could just get our neighbor-searching done outside the `jit`'d function and pass the results, but 
1. We would have to store a **huge** neighbor list of length $N_{\rm ngb} N$. This is often impractical for real-world datasets. 
2. Sometimes the demands of the algorithm don't allow this decoupling of operations, and we'd really like to switch gears into regular python on demand.

I know 2 ways of doing this. First, we can pass `forceobj=True` and code that cannot be JIT'd will run in object mode.


```python
from scipy.spatial import KDTree

tree = KDTree(coordinates)

@jit(forceobj=True)
def nbody_accel_nearest(masses, coordinates, num_neighbors, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses, accounting only for the num_neighbors set of nearest neighbors.

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
    accel = np.zeros_like(coordinates)  # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0]  # number of particles

    for i in range(N):
        _, ngb = tree.query(coordinates[i],num_neighbors+1)
        
        for j in ngb[1:]:
            # first need to calculate the distance between i and j
            distance = 0.0
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                distance += dx * dx
            if distance == 0:
                continue  # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                accel[i, k] += G * masses[j] * dx / distance**3 # computes the actual summand

    return accel
```


```python
%timeit nbody_accel_nearest(masses, coordinates, 16)
```

    /tmp/ipykernel_345291/1044257097.py:5: NumbaWarning: 
    Compilation is falling back to object mode WITHOUT looplifting enabled because Function "nbody_accel_nearest" failed type inference due to: Untyped global name 'tree': Cannot determine Numba type of <class 'scipy.spatial._kdtree.KDTree'>
    
    File "../../../../../tmp/ipykernel_345291/1044257097.py", line 32:
    <source missing, REPL/exec in use?>
    
      @jit(forceobj=True)


    119 ms ± 507 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)


We can see that this ran successfully, but kinda slow, especially considering that we are only evaluating $N_{\rm ngb} N$ forces now. Typically, embedding object-mode Python calls in loops that would otherwise be JIT'd will incur a performance cost.

An alternative is to use the [`objmode` context manager](https://numba.readthedocs.io/en/stable/user/withobjmode.html#numba.objmode):


```python
from numba import objmode, types

out_type = types.int64[:]
tree = KDTree(coordinates)

@jit
def nbody_accel_nearest(masses, coordinates, num_neighbors, G=1.):
    """
    Computes the Newtontian gravitational acceleration exerted on each of a set
    of point masses, accounting only for the num_neighbors set of nearest neighbors.

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
    accel = np.zeros_like(coordinates)  # array of zeros shaped like coordinate (N,3)
    N = coordinates.shape[0]  # number of particles

    for i in range(N):
        with objmode(ngb=out_type):
            _, ngb = tree.query(coordinates[i],num_neighbors+1)
        
        for j in ngb[1:]:
            # first need to calculate the distance between i and j
            distance = 0.0
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                distance += dx * dx
            if distance == 0:
                continue  # just skip if points lie on top of each other
            distance = np.sqrt(distance)

            # now compute the acceleration
            for k in range(3):
                dx = coordinates[j, k] - coordinates[i, k]
                accel[i, k] += G * masses[j] * dx / distance**3 # computes the actual summand

    return accel
```


```python
%timeit nbody_accel_nearest(masses, coordinates, 16)
```

    30.2 ms ± 195 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)


This was a little faster for some reason.

## Object-oriented programming with jitclasses

Numba supports JIT'd object-oriented code via the [jitclass](https://numba.readthedocs.io/en/stable/user/jitclass.html?highlight=jitclass). Again, only a subset of the full set of python programming features is available. Crucially, class inheritance is not current supported (last I checked), limiting the power of the approach. Nevertheless, it can still be quite useful to define data containers with associated methods.


```python
from numba.experimental import jitclass
from numba import float32 

spec = [
    ('x',float32),
    ('y',float32)
]

@jitclass(spec)
class Vector:
    """A crappy 2D vector class"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    @property
    def norm(self):
        """Returns the Euclidean norm of the vector"""
        return np.sqrt(self.x*self.x+self.y*self.y)    
    
    def __add__(self, other):
        """Returns the sum of the vector with another vector"""
        return Vector(self.x+other.x, self.y+other.y)
    
    def dot(self, other):
        return self.x*other.x + self.y*other.y
```


```python
vec= Vector(1,2)
vec.x, vec.y, vec.norm, (vec+vec).x, (vec+vec).y, vec.dot(vec)
```




    (1.0, 2.0, 2.2360680103302, 2.0, 4.0, 5.0)



# Other options for fast and parallel python

There are many other ways to write performant python code than just the numba parallel CPU coding we have done here.
- [numba also has aCUDA  API for running  Nvidia GPUs](https://numba.readthedocs.io/en/stable/cuda/index.html). Simple reductions and ufuncs are easy; more-complex custom kernels require some skill to get performance out of. As always with CUDA, **don't write your own kernel if a standard algorithm is available**
- Python has native [multiprocessing](https://docs.python.org/3/library/multiprocessing.html).
- [joblib](https://joblib.readthedocs.io/en/stable/) offers functionality for parallelism with a different implementation.
- [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) is best known for its use in machine learning but is also more broadly useful for coding GPU-portable python code consisting of function compositions and array operations.
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) implements MPI (Message Passing Interface) for distributed execution.

The best choice will depend on your particular problem and requirements.

Python may still have a reputation for being slow, but the reality is that in many cases the time-to-solution (coding+computation) for a give project can be shorter, just by taking advantage of Python's extensive community library support while optimizing the most numerically-intensive parts as we have here.
