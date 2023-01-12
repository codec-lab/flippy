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

def test_cps():
    check_cps_transform('''
    def fn(x):
        return x
    ''', '''
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    return x'
        return lambda : _cont(x)
    ''', check_args=[('a',)])

    check_cps_transform('''
    def fn(x):
        z = 0
        y = sum([1, 2, 3])
        x = x + 1
        z = z + 1
        return x + y + z
    ''', '''
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    z = 0\\n    y = sum([1, 2, 3])\\n    x = x + 1\\n    z = z + 1\\n    return x + y + z'
        z = 0
        _locals = locals()
        _scope_4 = {name: _locals[name] for name in ['x', 'z'] if name in _locals}

        def _cont_4(_res_4):
            if 'x' in _scope_4:
                x = _scope_4['x']
            if 'z' in _scope_4:
                z = _scope_4['z']

            y = _res_4
            x = x + 1
            z = z + 1
            return lambda : _cont(x + y + z)
        return lambda : _cps.interpret(sum, cont=_cont_4, stack=_stack, func_src=__func_src, locals_=_locals, lineno=4)([1, 2, 3])
    ''', check_args=[(0,), (1,), (2,)])

    check_cps_transform('''
    def fn(y):
        y = sum([y, 1])
        y = sum([y, 2])
        return y
    ''', '''
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    def fn(y, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(y):\\n    y = sum([y, 1])\\n    y = sum([y, 2])\\n    return y'
        _locals = locals()
        _scope_3 = {name: _locals[name] for name in ['y'] if name in _locals}

        def _cont_3(_res_3):
            if 'y' in _scope_3:
                y = _scope_3['y']
            y = _res_3
            _locals = locals()
            _scope_4 = {name: _locals[name] for name in ['y'] if name in _locals}

            def _cont_4(_res_4):
                if 'y' in _scope_4:
                    y = _scope_4['y']
                y = _res_4
                return lambda : _cont(y)
            return lambda : _cps.interpret(sum, cont=_cont_4, stack=_stack, func_src=__func_src, locals_=_locals, lineno=4)([y, 2])
        return lambda : _cps.interpret(sum, cont=_cont_3, stack=_stack, func_src=__func_src, locals_=_locals, lineno=3)([y, 1])
    ''', check_args=[(0,), (1,)])


def check_cps_transform(src, exp_src, *, check_args=[]):
    src = textwrap.dedent(src)
    node = ast.parse(src)
    node = CPSTransform()(node)

    exp_src = textwrap.dedent(exp_src)
    exp_node = ast.parse(exp_src)

    assert compare_ast(node, exp_node), f'Difference between transformed source and expected.\nExpected:\n{ast.unparse(exp_node)}\nTransformed:\n{ast.unparse(node)}'

    assert (
        isinstance(node, ast.Module) and
        len(node.body) == 1 and
        isinstance(node.body[0], ast.FunctionDef)
    )
    fn_name = node.body[0].name

    src_context = {}
    exec(src, src_context)

    exp_context = {}
    interpreter = CPSInterpreter()
    exp_context[CPSTransform.cps_interpreter_name] = interpreter
    exec(exp_src, exp_context)

    for args in check_args:
        # We execute using only a simple trampoline, so this implementation can't handle stochastic primitives.
        assert src_context[fn_name](*args) == trampoline(exp_context[fn_name](*args))

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