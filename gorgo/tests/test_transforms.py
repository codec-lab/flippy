import itertools
import textwrap
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

def test_desugaring_transform():
    src_compiled = [
        ("b = f(g(a))", "__v0 = g(a); b = f(__v0)"),
        (
            "c = 0 if a > 5 else 1",
            textwrap.dedent("""
            if a > 5:
                __v0 = 0
            else:
                __v0 = 1
            c = __v0
            """)
        ),
        (
            "d = (lambda x, y=1: g(x)*100)()",
            textwrap.dedent("""
            def __v0(x, y=1):
                __v1 = g(x)
                return __v1*100
            d = __v0()
            """)
        )
    ]
    for src, comp in src_compiled:
        node = ast.parse(src)
        node = DesugaringTransform()(node)
        assert compare_ast(node, ast.parse(comp))

def compare_ast(node1, node2):
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ('lineno', 'col_offset', 'ctx', 'end_col_offset', 'end_lineno', '_parent'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, itertools.zip_longest(node1, node2)))
    else:
        return node1 == node2