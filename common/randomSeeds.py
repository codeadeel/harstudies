# This file is responsible for fixing the random seeds and native thread limits of every run
# %%
# Importing Libraries
import random

import numpy as np
from threadpoolctl import threadpool_limits


# %%
# Seeding Helpers
def seedEverything(randomSeed):
    """
    This function seeds the python and numpy random generators

    Arguments
    =========
    randomSeed : Seed for every random operation

    Output
    ======
    Numpy random generator created from the same seed
    """
    # Seed The Global Generators
    random.seed(randomSeed)
    np.random.seed(randomSeed)
    return np.random.default_rng(randomSeed)


def limitThreads(nJobs):
    """
    This function limits the BLAS and OpenMP thread pools that are already loaded to the configured job count

    Arguments
    =========
    nJobs : Number of threads each native library may use

    Output
    ======
    Thread limit controller that can restore the original limits
    """
    # Limit The Native Thread Pools
    return threadpool_limits(limits=nJobs)
