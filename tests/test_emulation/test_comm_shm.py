import threading
import time
from multiprocessing.shared_memory import SharedMemory

from morphling.emulation.barrier import StructuralBarrier
from morphling.emulation.comm_shm import ShmAllReduceCommFn


def _run_comm(comm_fn: ShmAllReduceCommFn) -> None:
    comm_fn()


def test_shm_allreduce_2_devices():
    barrier = StructuralBarrier(2)
    comm0 = ShmAllReduceCommFn(
        rank=0,
        world_size=2,
        tensor_size=128,
        barrier=barrier,
    )
    comm1 = ShmAllReduceCommFn(
        rank=1,
        world_size=2,
        tensor_size=128,
        barrier=barrier,
    )

    t0 = threading.Thread(target=_run_comm, args=(comm0,))
    t1 = threading.Thread(target=_run_comm, args=(comm1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    assert comm0.last_result is not None
    assert comm1.last_result is not None
    assert comm0.last_result.shape[0] == 128
    assert comm1.last_result.shape[0] == 128


def test_shm_wall_time_nonzero():
    barrier = StructuralBarrier(2)
    comm0 = ShmAllReduceCommFn(
        rank=0,
        world_size=2,
        tensor_size=256,
        barrier=barrier,
    )
    comm1 = ShmAllReduceCommFn(
        rank=1,
        world_size=2,
        tensor_size=256,
        barrier=barrier,
    )

    t0 = threading.Thread(target=_run_comm, args=(comm0,))
    t1 = threading.Thread(target=_run_comm, args=(comm1,))
    start = time.perf_counter()
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    elapsed = time.perf_counter() - start

    assert elapsed > 0


def test_shm_cleanup():
    barrier = StructuralBarrier(2)
    comm0 = ShmAllReduceCommFn(
        rank=0,
        world_size=2,
        tensor_size=64,
        barrier=barrier,
    )
    comm1 = ShmAllReduceCommFn(
        rank=1,
        world_size=2,
        tensor_size=64,
        barrier=barrier,
    )

    t0 = threading.Thread(target=_run_comm, args=(comm0,))
    t1 = threading.Thread(target=_run_comm, args=(comm1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    missing = 0
    for name in (
        "emulation_grad_0_0",
        "emulation_grad_0_1",
        "emulation_result_0",
    ):
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
        except FileNotFoundError:
            missing += 1
    assert missing == 3
