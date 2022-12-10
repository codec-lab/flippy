from gorgo.transforms import *
from gorgo.interpreter import CPSInterpreter
from gorgo import keep_deterministic

def trampoline(thunk):
    while callable(thunk):
        thunk = thunk()
    return thunk

def interpret(func, *args, **kwargs):
    interpreter = CPSInterpreter()
    local_context = {**interpreter.get_closure(func), "_cps": interpreter}
    print(func.__name__, local_context)
    code = interpreter.transform_from_func(func)
    print(code)
    exec(ast.unparse(code), func.__globals__, local_context)
    trans_func = local_context[func.__name__]
    return trampoline(trans_func(*args, **kwargs))

def helper_in_module(x):
    return x ** 2

def main_in_module_helper_in_module():
    return helper_in_module(3)

def main_in_module_helper_in_closure():
    def helper_in_closure(x):
        return x ** 2
    return helper_in_closure(3)

def test_x():
    def helper_in_function(x):
        return x ** 2
    def main_in_function_helper_in_function():
        return helper_in_function(3)

    def main_in_function_helper_in_closure():
        def helper_in_closure(x):
            return x ** 2
        return helper_in_closure(3)

    assert interpret(main_in_module_helper_in_closure) == 9
    assert interpret(main_in_module_helper_in_module) == 9
    assert interpret(main_in_function_helper_in_closure) == 9
    # TODO: fix!
    # assert interpret(main_in_function_helper_in_function) == 9
