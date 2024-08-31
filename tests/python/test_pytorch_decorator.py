import torch


def torch_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("torch_decorator ", funcname)
        return func(*args, **kwargs)

    return wrapper


a = torch.ones((1, 3))
b = torch.ones((1, 3))


pyobj_keywords = [
    "__class__",
    "__delattr__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__module__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__weakref__",
]

# for all functions in torch.Tensor, add decorator
functions = [f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))]
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(torch.Tensor, f)
    setattr(torch.Tensor, f, torch_decorator(func, f))

# for all functions in torch, add decorator
functions = [f for f in dir(torch) if callable(getattr(torch, f))]
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(torch, f)
    setattr(torch, f, torch_decorator(func, f))

print(dir(a))

c = a @ b.T
c = a + 1