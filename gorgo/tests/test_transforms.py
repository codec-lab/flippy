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
        ("b = f(g(a))", "__v0 = g(a); __v1 = f(__v0); b = __v1"),
        (
            "a = lambda x: g(h(x))",
            textwrap.dedent("""
            def __v0(x):
                __v1 = h(x)
                __v2 = g(__v1)
                return __v2
            a = __v0
            """)
        ),
        (
            "a = g(h(x)) if f(x) else h(g(x))",
            textwrap.dedent("""
            __v2 = f(x)
            __v0 = __v2
            if __v0:
                __v3 = h(x)
                __v4 = g(__v3)
                __v1 = __v4
            else:
                __v5 = g(x)
                __v6 = h(__v5)
                __v1 = __v6
            a = __v1
            """)
        ),
        (
            "d = (lambda x, y=1: g(x)*100)()",
            textwrap.dedent("""
            def __v0(x, y=1):
                __v1 = g(x)
                return __v1*100
            __v2 = __v0()
            d = __v2
            """)
        ),
        (
            textwrap.dedent("""
            def f():
                pass
            """),
            textwrap.dedent("""
            def f():
                pass
                return None
            """)
        )
    ]
    for src, comp in src_compiled:
        node = ast.parse(src)
        node = DesugaringTransform()(node)
        assert compare_ast(node, ast.parse(comp)), src

def compare_sourcecode_to_equivalent_sourcecode(src, exp_src):
    node = ast.parse(src)
    targ_node = ast.parse(exp_src)
    node = DesugaringTransform()(node)
    targ_node = DesugaringTransform()(targ_node)
    assert compare_ast(node, targ_node)

    src_context = {}
    exec(src, src_context)
    exp_context = {}
    exec(exp_src, exp_context)
    for args in [(1, 2, 3), ('b', 'a', ''), (True, False, True)]:
        assert src_context['f'](*args) == exp_context['f'](*args)
        
def test_multiple_and_transform():
    src = textwrap.dedent("""
    def f(a, b, c):
        return a and b and c
    """)
    exp_src = textwrap.dedent("""
    def f(a, b, c):
        return ((a and b) and c)
    """)
    compare_sourcecode_to_equivalent_sourcecode(src, exp_src)
    
def test_multiple_or_transform():
    src = textwrap.dedent("""
    def f(a, b, c):
        return a or b or c
    """)
    exp_src = textwrap.dedent("""
    def f(a, b, c):
        return ((a or b) or c)
    """)
    compare_sourcecode_to_equivalent_sourcecode(src, exp_src)

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