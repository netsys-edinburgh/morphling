"""Demo: globally decorate every torch / torch.Tensor callable.

NOT a pytest test. This script monkey-patches torch.* and torch.Tensor.*
at module import time. Running it inside the pytest process corrupts
torch for every subsequently collected test (isinstance checks fail,
torch.cuda lazy init throws ``registerOp not callable``, etc.).

The leading underscore in the filename keeps pytest from auto-collecting
it. Run it directly::

    python3 tests/python/unit/_demo_pytorch_decorator.py
"""

import torch

from morphling.common.keywords import pyobj_keywords, pytorch_keywords


def torch_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("torch_decorator ", funcname)
        return func(*args, **kwargs)

    return wrapper


a = torch.ones((1, 3))
b = torch.ones((1, 3))


# for all functions in torch.Tensor, add decorator
functions = [f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))]
for f in functions:
    if f in pyobj_keywords or f in pytorch_keywords:
        continue
    func = getattr(torch.Tensor, f)
    setattr(torch.Tensor, f, torch_decorator(func, f))

# for all functions in torch, add decorator
functions = [f for f in dir(torch) if callable(getattr(torch, f))]
for f in functions:
    if f in pyobj_keywords or f in pytorch_keywords:
        continue
    func = getattr(torch, f)
    setattr(torch, f, torch_decorator(func, f))

print(dir(a))

c = a @ b.T
c = a + 1
