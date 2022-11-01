# The profile decorator requires running under
# https://pypi.org/project/memory-profiler/ via `mprof run
# impl_hdf5_transfer_data_bench.py`

import h5py
import numpy as np

DIM = 1000
NDIM = 3
DSET_NAME = "x"

# x = np.random.random(size=[dim for _ in range(ndim)])

FNAME_TO = "transfer_to.h5"
FNAME_FROM = "transfer_from.h5"


@profile
def generate() -> None:
    with h5py.File(FNAME_FROM, "w") as file_from:
        # file_from[DSET_NAME] = np.random.random(size=[DIM for _ in range(NDIM)])
        dset = file_from.create_dataset(
            name=DSET_NAME,
            data=np.random.random(size=[DIM for _ in range(NDIM)]),
            compression="gzip",
            chunks=tuple(100 for _ in range(NDIM)),
        )


@profile
def transfer() -> None:
    with h5py.File(FNAME_TO, "w") as file_to, h5py.File(FNAME_FROM, "r") as file_from:
        # file_to[dim_name] = file_from lib import funs[dim_name]
        file_from.copy(source=file_from[DSET_NAME], dest=file_to)


@profile
def transform() -> None:
    # can't transform during copy, so do it in-place
    with h5py.File(FNAME_TO, "r+") as file_to:
        dset = file_to[DSET_NAME]
        for s in dset.iter_chunks():
            dset[s] += 2500.0


generate()
transfer()
transform()
