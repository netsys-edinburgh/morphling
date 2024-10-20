def func_access_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("func_access ", funcname)
        return func(*args, **kwargs)

    return wrapper
