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
    context = {**func.__globals__, **interpreter.get_closure(func), "_cps": interpreter}
    print(func.__name__, context)
    code = interpreter.transform_from_func(func)
    print(code)
    exec(ast.unparse(code), context)
    trans_func = context[func.__name__]
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
    assert interpret(main_in_function_helper_in_function) == 9

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
            def __v0():
                __v3 = f(x)
                __v1 = __v3
                if __v1:
                    __v4 = h(x)
                    __v5 = g(__v4)
                    __v2 = __v5
                else:
                    __v6 = g(x)
                    __v7 = h(__v6)
                    __v2 = __v7
                return __v2
            __v8 = __v0()
            a = __v8
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
        ),

        # Various cases for assignment
        ('x += 123', 'x = x + 123'),
        ('x.y += 123', 'x.y = x.y + 123'),
        ('x[0] += 123', 'x[0] = x[0] + 123'),
        ('x[0].prop += 123', 'x[0].prop = x[0].prop + 123'),
        ('x: int = 123', 'x = 123'),
        ('x: int', ''),

        # List Comprehensions
        (
            '[x for x in range(3)]',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                return __acc + [x]
            __v1 = range(3)
            __v2 = recursive_reduce(__v0, __v1, [])
            __v2
            '''),
        ),
        (
            # Only one test
            '[x for x in range(3) if x]',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                def __v1():
                    __v2 = x
                    if __v2:
                        __v3 = __acc + [x]
                    else:
                        __v3 = __acc
                    return __v3
                __v4 = __v1()
                return __v4
            __v5 = range(3)
            __v6 = recursive_reduce(__v0, __v5, [])
            __v6
            '''),
        ),
        (
            # Two tests
            '[x for x in range(3) if x if x**2]',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                def __v1():
                    __v4 = x
                    if __v4:
                        __v5 = x ** 2
                    else:
                        __v5 = __v4
                    __v2 = __v5
                    if __v2:
                        __v3 = __acc + [x]
                    else:
                        __v3 = __acc
                    return __v3
                __v6 = __v1()
                return __v6
            __v7 = range(3)
            __v8 = recursive_reduce(__v0, __v7, [])
            __v8
            '''),
        ),
        (
            '[(x, y) for x in range(3) if x for y in range(4)]',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v1(__acc, __target):
                x = __target
                def __v0(__acc, __target):
                    y = __target
                    return __acc + [(x, y)]
                def __v2():
                    __v3 = x
                    if __v3:
                        __v5 = range(4)
                        __v6 = recursive_reduce(__v0, __v5, [])
                        __v4 = __acc + __v6
                    else:
                        __v4 = __acc
                    return __v4
                __v7 = __v2()
                return __v7
            __v8 = range(3)
            __v9 = recursive_reduce(__v1, __v8, [])
            __v9
            '''),
        ),

        # Set/Dict Comprehensions
        (
            '{x for x in range(3)}',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                return __acc | {x}
            __v1 = range(3)
            __v2 = set()
            __v3 = recursive_reduce(__v0, __v1, __v2)
            __v3
            '''),
        ),
        (
            '{x: x**2 for x in range(3)}',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                return __acc | {x: x**2}
            __v1 = range(3)
            __v2 = recursive_reduce(__v0, __v1, {})
            __v2
            '''),
        ),
        (
            '{x: y**2 for x, y in {}.items()}',
            textwrap.dedent('''
            from gorgo import recursive_reduce
            def __v0(__acc, __target):
                (x, y) = __target
                return __acc | {x: y**2}
            __v1 = {}.items()
            __v2 = recursive_reduce(__v0, __v1, {})
            __v2
            '''),
        ),

        # de-decoration
        (
            textwrap.dedent("""
            @decorator
            def f():
                return 1
            """),
            textwrap.dedent("""
            def f():
                return 1
            __v0 = decorator(f)
            f = __v0
            """)
        )
    ]

    for src, comp in src_compiled:
        node = ast.parse(src)
        node = DesugaringTransform()(node)
        assert compare_ast(node, ast.parse(comp)), f'expected:\n{comp}\nfound:\n{ast.unparse(node)}'

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
    # Basic case of CPS.
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

    # Basic test of names defined by function arguments + assignment
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
        _locals_4 = locals()
        _scope_4 = {name: _locals_4[name] for name in ['x', 'z'] if name in _locals_4}

        def _cont_4(_res_4):
            if 'x' in _scope_4:
                x = _scope_4['x']
            if 'z' in _scope_4:
                z = _scope_4['z']

            y = _res_4
            x = x + 1
            z = z + 1
            return lambda : _cont(x + y + z)
        return lambda : _cps.interpret(sum, cont=_cont_4, stack=_stack, func_src=__func_src, locals_=_locals_4, lineno=4)([1, 2, 3])
    ''', check_args=[(0,), (1,), (2,)])

    # Making sure things still work well in nested continuations.
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
        _locals_3 = locals()
        _scope_3 = {name: _locals_3[name] for name in ['y'] if name in _locals_3}

        def _cont_3(_res_3):
            if 'y' in _scope_3:
                y = _scope_3['y']
            y = _res_3
            _locals_4 = locals()
            _scope_4 = {name: _locals_4[name] for name in ['y'] if name in _locals_4}

            def _cont_4(_res_4):
                if 'y' in _scope_4:
                    y = _scope_4['y']
                y = _res_4
                return lambda : _cont(y)
            return lambda : _cps.interpret(sum, cont=_cont_4, stack=_stack, func_src=__func_src, locals_=_locals_4, lineno=4)([y, 2])
        return lambda : _cps.interpret(sum, cont=_cont_3, stack=_stack, func_src=__func_src, locals_=_locals_3, lineno=3)([y, 1])
    ''', check_args=[(0,), (1,)])

    # Testing destructuring.
    check_cps_transform('''
    def fn(x):
        [y, z] = x
        sum([])
        return y + z
    ''', '''
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    @lambda fn: (fn, setattr(fn, '_cps_transformed', True))[0]
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    [y, z] = x\\n    sum([])\\n    return y + z'
        [y, z] = x
        _locals_4 = locals()
        _scope_4 = {name: _locals_4[name] for name in ['x', 'y', 'z'] if name in _locals_4}

        def _cont_4(_res_4):
            if 'x' in _scope_4:
                x = _scope_4['x']
            if 'y' in _scope_4:
                y = _scope_4['y']
            if 'z' in _scope_4:
                z = _scope_4['z']
            _res_4
            return lambda : _cont(y + z)
        return lambda : _cps.interpret(sum, cont=_cont_4, stack=_stack, func_src=__func_src, locals_=_locals_4, lineno=4)([])
    ''', check_args=[([1, 2],), ([7, 3],)])

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
            if k in ('lineno', 'col_offset', 'end_col_offset', 'end_lineno'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, itertools.zip_longest(node1, node2)))
    else:
        return node1 == node2
