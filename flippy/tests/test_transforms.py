import itertools
import textwrap
from flippy.transforms import *
from flippy.interpreter import CPSInterpreter
from flippy.hashable import hashablelist, hashabledict
from flippy import keep_deterministic

def trampoline(thunk):
    while callable(thunk):
        thunk = thunk()
    return thunk

def interpret(func, *args, **kwargs):
    interpreter = CPSInterpreter()
    context = {
        **func.__globals__,
        **interpreter.get_closure(func),
        "_cps": interpreter,
        "hashablelist": hashablelist,
        "hashabledict": hashabledict,
    }
    print(func.__name__, context)
    code = interpreter.transform_from_func(func)
    print(code)
    exec(ast.unparse(code), context)
    trans_func = context[func.__name__]
    return trampoline(trans_func(*args, **kwargs))

class ScopeTools:
    @classmethod
    def pack(cls, lineno, names, *, hashabledict=False):
        constructor = 'hashabledict' if hashabledict else ''
        import_ = 'from flippy.hashable import hashabledict' if hashabledict else ''
        return ast.parse(textwrap.dedent(f'''
            {import_}
            _locals_{lineno} = locals()
            _scope_{lineno} = {constructor}({{name: _locals_{lineno}[name] for name in {names} if name in _locals_{lineno}}})
        ''')).body
    @classmethod
    def unpack(cls, lineno, names):
        return ast.parse(textwrap.dedent('\n'.join(f'''
            if '{n}' in _scope_{lineno}:
                {n} = _scope_{lineno}['{n}']
        ''' for n in names))).body
    @classmethod
    def pack_unpack(cls, lineno, names, *, tag=None, hashabledict=False):
        if tag is None:
            tag = f'_{lineno}'
        return {
            f'pack{tag}': cls.pack(lineno, names, hashabledict=hashabledict),
            f'unpack{tag}': cls.unpack(lineno, names),
        }
    def fill(code, replacements):
        return ast.unparse(Placeholder.fill(ast.parse(textwrap.dedent(code)), replacements))

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
        ("b = f(g(a))", "b = f(g(a))"),
        (
            "a = lambda x: g(h(x))",
            textwrap.dedent("""
            def __v0(x):
                return g(h(x))
            a = __v0
            """)
        ),
        (
            "a = g(h(x)) if f(x) else h(g(x))",
            textwrap.dedent("""
            def __v0():
                __v1 = f(x)
                if __v1:
                    __v2 = g(h(x))
                else:
                    __v2 = h(g(x))
                return __v2
            a = __v0()
            """)
        ),
        (
            "d = (lambda x, y=1: g(x)*100)()",
            textwrap.dedent("""
            def __v0(x, y=1):
                return g(x)*100
            d = __v0()
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

        # Loops
        (
            textwrap.dedent('''
            while a:
                b
            '''),
            textwrap.dedent('''
            while True:
                if not a:
                    break
                b
            '''),
        ),
        (
            textwrap.dedent('''
            for x in range(3):
                x
            '''),
            textwrap.dedent('''
            _for_iter_2 = range(3)
            _for_idx_2 = 0
            while True:
                if not _for_idx_2 < len(_for_iter_2):
                    break
                x = _for_iter_2[_for_idx_2]
                _for_idx_2 = _for_idx_2 + 1
                x
            '''),
        ),

        # List Comprehensions
        (
            '[x for x in range(3)]',
            textwrap.dedent('''
            from flippy import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                return __acc + [x]
            recursive_reduce(__v0, range(3), [])
            '''),
        ),
        (
            # Only one test
            '[x for x in range(3) if x]',
            textwrap.dedent('''
            from flippy import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                def __v1():
                    __v2 = x
                    if __v2:
                        __v3 = __acc + [x]
                    else:
                        __v3 = __acc
                    return __v3
                return __v1()
            recursive_reduce(__v0, range(3), [])
            '''),
        ),
        (
            # Two tests
            '[x for x in range(3) if x if x**2]',
            textwrap.dedent('''
            from flippy import recursive_reduce
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
                return __v1()
            recursive_reduce(__v0, range(3), [])
            '''),
        ),
        (
            '[(x, y) for x in range(3) if x for y in range(4)]',
            textwrap.dedent('''
            from flippy import recursive_reduce
            def __v1(__acc, __target):
                x = __target
                def __v0(__acc, __target):
                    y = __target
                    return __acc + [(x, y)]
                def __v2():
                    __v3 = x
                    if __v3:
                        __v4 = __acc + recursive_reduce(__v0, range(4), [])
                    else:
                        __v4 = __acc
                    return __v4
                return __v2()
            recursive_reduce(__v1, range(3), [])
            '''),
        ),

        # Set/Dict Comprehensions
        (
            '{x for x in range(3)}',
            textwrap.dedent('''
            from flippy import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                return __acc | {x}
            recursive_reduce(__v0, range(3), set())
            '''),
        ),
        (
            '{x: x**2 for x in range(3)}',
            textwrap.dedent('''
            from flippy import recursive_reduce
            def __v0(__acc, __target):
                x = __target
                return __acc | {x: x**2}
            recursive_reduce(__v0, range(3), {})
            '''),
        ),
        (
            '{x: y**2 for x, y in {}.items()}',
            textwrap.dedent('''
            from flippy import recursive_reduce
            def __v0(__acc, __target):
                (x, y) = __target
                return __acc | {x: y**2}
            recursive_reduce(__v0, {}.items(), {})
            '''),
        ),

    ]

    for src, comp in src_compiled:
        node = ast.parse(src)
        node = DesugaringTransform()(node)
        assert_compare_ast(node, ast.parse(comp))

def compare_sourcecode_to_equivalent_sourcecode(src, exp_src):
    node = ast.parse(src)
    targ_node = ast.parse(exp_src)
    node = DesugaringTransform()(node)
    targ_node = DesugaringTransform()(targ_node)
    assert_compare_ast(targ_node, node)

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

def test_cps_call():
    # Basic case of CPS.
    check_cps_transform('''
    def fn(x):
        return x
    ''', '''
    @lambda fn: CPSFunction(fn, 'def fn(x):\\n    return x')
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
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn(x):\\n    z = 0\\n    y = sum([1, 2, 3])\\n    x = x + 1\\n    z = z + 1\\n    return x + y + z')
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    z = 0\\n    y = sum([1, 2, 3])\\n    x = x + 1\\n    z = z + 1\\n    return x + y + z'
        z = 0
        {Placeholder.new('pack_0')}

        def _cont_0(_res_0):
            {Placeholder.new('unpack_0')}

            y = _res_0
            x = x + 1
            z = z + 1
            return lambda : _cont(x + y + z)
        return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=2)([1, 2, 3])
    ''', ScopeTools.pack_unpack(0, ['x', 'z'])), check_args=[(0,), (1,), (2,)])

    # Making sure things still work well in nested continuations.
    check_cps_transform('''
    def fn(y):
        y = sum([y, 1])
        y = sum([y, 2])
        return y
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn(y):\\n    y = sum([y, 1])\\n    y = sum([y, 2])\\n    return y')
    def fn(y, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(y):\\n    y = sum([y, 1])\\n    y = sum([y, 2])\\n    return y'
        {Placeholder.new('pack_0')}

        def _cont_0(_res_0):
            {Placeholder.new('unpack_0')}
            y = _res_0
            {Placeholder.new('pack_1')}

            def _cont_1(_res_1):
                {Placeholder.new('unpack_1')}
                y = _res_1
                return lambda : _cont(y)
            return lambda : _cps.interpret(sum, cont=_cont_1, stack=_stack, func_src=__func_src, locals_=_locals_1, call_id=1, lineno=2)([y, 2])
        return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=1)([y, 1])
    ''', {
        **ScopeTools.pack_unpack(0, ['y']),
        **ScopeTools.pack_unpack(1, ['y']),
    }), check_args=[(0,), (1,)])

    # Testing destructuring.
    check_cps_transform('''
    def fn(x):
        [y, z] = x
        sum([])
        return y + z
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn(x):\\n    [y, z] = x\\n    sum([])\\n    return y + z')
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    [y, z] = x\\n    sum([])\\n    return y + z'
        [y, z] = x
        {Placeholder.new('pack_0')}

        def _cont_0(_res_0):
            {Placeholder.new('unpack_0')}
            _res_0
            return lambda : _cont(y + z)
        return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=2)([])
    ''', {
        **ScopeTools.pack_unpack(0, ['x', 'y', 'z']),
    }), check_args=[([1, 2],), ([7, 3],)])

def test_cps_new_vars_scope_packing():
    # checking that new variables are added at the appropriate time
    check_cps_transform('''
    def fn():
        x = 3
        sum([])
        y = 4
        sum([])
        z = 5
        return x + y + z
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn():\\n    x = 3\\n    sum([])\\n    y = 4\\n    sum([])\\n    z = 5\\n    return x + y + z')
    def fn(*, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn():\\n    x = 3\\n    sum([])\\n    y = 4\\n    sum([])\\n    z = 5\\n    return x + y + z'
        x = 3
        {Placeholder.new('pack_0')}

        def _cont_0(_res_0):
            {Placeholder.new('unpack_0')}
            _res_0
            y = 4
            {Placeholder.new('pack_1')}

            def _cont_1(_res_1):
                {Placeholder.new('unpack_1')}
                _res_1
                z = 5
                return lambda : _cont(x + y + z)
            return lambda : _cps.interpret(sum, cont=_cont_1, stack=_stack, func_src=__func_src, locals_=_locals_1, call_id=1, lineno=4)([])
        return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=2)([])
    ''', {
        **ScopeTools.pack_unpack(0, ['x']),
        **ScopeTools.pack_unpack(1, ['x', 'y']),
    }), check_args=[()])

def test_cps_desugared_calls():
    check_cps_transform('''
    def fn():
        __v0 = sum([2, 3])
        __v1 = sum([1, __v0])
        return __v1
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn():\\n    __v0 = sum([2, 3])\\n    __v1 = sum([1, __v0])\\n    return __v1')
    def fn(*, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn():\\n    __v0 = sum([2, 3])\\n    __v1 = sum([1, __v0])\\n    return __v1'
        {Placeholder.new('pack_0')}

        def _cont_0(_res_0):
            pass
            __v0 = _res_0
            {Placeholder.new('pack_1')}

            def _cont_1(_res_1):
                {Placeholder.new('unpack_1')}
                __v1 = _res_1
                return lambda : _cont(__v1)
            return lambda : _cps.interpret(sum, cont=_cont_1, stack=_stack, func_src=__func_src, locals_=_locals_1, call_id=1, lineno=2)([1, __v0])
        return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=1)([2, 3])
    ''', {
        **ScopeTools.pack_unpack(0, []),
        **ScopeTools.pack_unpack(1, ['__v0']),
    }), check_args=[()])

def test_cps_if():
    check_cps_transform('''
    def fn(x):
        if x == 0:
            y = 0
        else:
            y = 1
        return y * 2
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn(x):\\n    if x == 0:\\n        y = 0\\n    else:\\n        y = 1\\n    return y * 2')
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    if x == 0:\\n        y = 0\\n    else:\\n        y = 1\\n    return y * 2'
        def _cont_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont(y * 2)
        if x == 0:
            y = 0
            {Placeholder.new('pack_0')}
            return lambda : _cont_0(_scope_0)
        else:
            y = 1
            {Placeholder.new('pack_0')}
            return lambda : _cont_0(_scope_0)
    ''', {
        **ScopeTools.pack_unpack(0, ['x', 'y'])
    }), check_args=[(0,), (1,)])

def test_cps_if_nested_call():
    check_cps_transform('''
    def fn(x):
        if x == 0:
            y = 0
        else:
            y = sum([3, x])
        return y * 2
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn(x):\\n    if x == 0:\\n        y = 0\\n    else:\\n        y = sum([3, x])\\n    return y * 2')
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn(x):\\n    if x == 0:\\n        y = 0\\n    else:\\n        y = sum([3, x])\\n    return y * 2'

        def _cont_1(_scope_1):
            {Placeholder.new('unpack_1')}
            return lambda : _cont(y * 2)
        if x == 0:
            y = 0
            {Placeholder.new('pack_1')}
            return lambda : _cont_1(_scope_1)
        else:
            {Placeholder.new('pack_0')}

            def _cont_0(_res_0):
                {Placeholder.new('unpack_0')}
                y = _res_0
                {Placeholder.new('pack_1')}
                return lambda : _cont_1(_scope_1)
            return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=4)([3, x])
    ''', {
        **ScopeTools.pack_unpack(0, ['x', 'y']),
        **ScopeTools.pack_unpack(1, ['x', 'y']),
    }), check_args=[(0,), (1,)])


def test_cps_if_call_in_node_test():
    check_cps_transform('''
    def fn(x):
        if 0 == sum(x):
            y = 0
        else:
            y = 1
        return y * 2
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fn(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        {Placeholder.new('pack_0')}

        def _cont_0(_res_0):
            {Placeholder.new('unpack_0')}
            def _cont_1(_scope_1):
                {Placeholder.new('unpack_1')}
                return lambda : _cont(y * 2)
            if 0 == _res_0:
                y = 0
                {Placeholder.new('pack_1')}
                return lambda : _cont_1(_scope_1)
            else:
                y = 1
                {Placeholder.new('pack_1')}
                return lambda : _cont_1(_scope_1)
        return lambda : _cps.interpret(sum, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=1)(x)
    ''', {
        **ScopeTools.pack_unpack(0, ['x']),
        **ScopeTools.pack_unpack(1, ['x', 'y']),
    }), check_args=[([-1, 1],), ([3],)])


def test_cps_while_simple():
    check_cps_transform('''
    def fib(x):
        a, b = 1, 1
        ct = 0
        while ct < x:
            a, b = b, a + b
            ct = ct + 1
        return a
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fib(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        (a, b) = (1, 1)
        ct = 0
        @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
        def _loop_fn_0(_scope_0, *, _stack=(), _cps=_cps, _cont=lambda val: val):
            __func_src = "$IGNORE_STRING"
            {Placeholder.new('unpack_0')}
            if ct < x:
                (a, b) = (b, a + b)
                ct = ct + 1
                {Placeholder.new('pack_0')}
                return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
            else:
                {Placeholder.new('pack_0')}
                return lambda : _cont(_scope_0)

        def _loop_exit_fn_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont(a)
        {Placeholder.new('pack_0')}
        return lambda : _cps.interpret(_loop_fn_0, cont=_loop_exit_fn_0, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
    ''', {
        **ScopeTools.pack_unpack(0, ['a', 'b', 'ct', 'x'], hashabledict=True)
    }), check_args=[(i,) for i in range(10)])


def test_cps_while_desugared():
    check_cps_transform('''
    def fib(x):
        a, b = 1, 1
        ct = 0
        while ct < x:
            a, b = b, a + b
            ct = ct + 1
        return a
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fib(x, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        (a, b) = (1, 1)
        ct = 0
        @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
        def _loop_fn_0(_scope_0, *, _stack=(), _cps=_cps, _cont=lambda val: val):
            __func_src = "$IGNORE_STRING"
            {Placeholder.new('unpack_0')}
            if True:
                def _cont_1(_scope_1):
                    {Placeholder.new('unpack_1')}
                    (a, b) = (b, a + b)
                    ct = ct + 1
                    {Placeholder.new('pack_0')}
                    return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
                if not ct < x:
                    {Placeholder.new('pack_0')}
                    return lambda : _cont(_scope_0)
                    {Placeholder.new('pack_1')}
                    return lambda : _cont_1(_scope_1)
                else:
                    {Placeholder.new('pack_1')}
                    return lambda : _cont_1(_scope_1)
            else:
                {Placeholder.new('pack_0')}
                return lambda : _cont(_scope_0)

        def _loop_exit_fn_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont(a)
        {Placeholder.new('pack_0')}
        return lambda : _cps.interpret(_loop_fn_0, cont=_loop_exit_fn_0, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
    ''', {
        **ScopeTools.pack_unpack(0, ['a', 'b', 'ct', 'x'], hashabledict=True),
        **ScopeTools.pack_unpack(1, ['a', 'b', 'ct', 'x']),
    }), check_args=[(i,) for i in range(10)], desugar=True)


def test_cps_while_continue_break():
    check_cps_transform('''
    def fn():
        rv = []
        ct = 0
        while ct < 10:
            ct = ct + 1
            if ct == 7:
                break
            if ct == 3:
                continue
            rv = rv + [ct]
        return rv
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fn(*, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        rv = []
        ct = 0

        @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
        def _loop_fn_0(_scope_0, *, _stack=(), _cps=_cps, _cont=lambda val: val):
            __func_src = "$IGNORE_STRING"
            {Placeholder.new('unpack_0')}
            if ct < 10:
                ct = ct + 1

                def _cont_1(_scope_1):
                    {Placeholder.new('unpack_1')}

                    def _cont_2(_scope_2):
                        {Placeholder.new('unpack_2')}
                        rv = rv + [ct]
                        {Placeholder.new('pack_0')}
                        return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
                    if ct == 3:
                        {Placeholder.new('pack_0')}
                        return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
                        {Placeholder.new('pack_2')}
                        return lambda : _cont_2(_scope_2)
                    else:
                        {Placeholder.new('pack_2')}
                        return lambda : _cont_2(_scope_2)
                if ct == 7:
                    {Placeholder.new('pack_0')}
                    return lambda : _cont(_scope_0)
                    {Placeholder.new('pack_1')}
                    return lambda : _cont_1(_scope_1)
                else:
                    {Placeholder.new('pack_1')}
                    return lambda : _cont_1(_scope_1)
            else:
                {Placeholder.new('pack_0')}
                return lambda : _cont(_scope_0)

        def _loop_exit_fn_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont(rv)
        {Placeholder.new('pack_0')}
        return lambda : _cps.interpret(_loop_fn_0, cont=_loop_exit_fn_0, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
    ''', {
        **ScopeTools.pack_unpack(0, ['ct', 'rv'], hashabledict=True),
        **ScopeTools.pack_unpack(1, ['ct', 'rv']),
        **ScopeTools.pack_unpack(2, ['ct', 'rv']),
    }), check_args=[()], check_out=[[1, 2, 4, 5, 6]])


def test_cps_while_nested():
    names = ['lim', 'rv', 'x', 'y']
    exp = ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fn(lim, *, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        rv = []
        x = 0

        @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
        def _loop_fn_0(_scope_0, *, _stack=(), _cps=_cps, _cont=lambda val: val):
            __func_src = "$IGNORE_STRING"
            {Placeholder.new('unpack_0')}
            if x < lim:
                y = 0

                @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
                def _loop_fn_1(_scope_1, *, _stack=(), _cps=_cps, _cont=lambda val: val):
                    __func_src = "$IGNORE_STRING"
                    {Placeholder.new('unpack_1')}
                    if y < x:
                        rv += [(x, y)]
                        y = y + 1
                        {Placeholder.new('pack_1')}
                        return lambda : _cps.interpret(_loop_fn_1, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_1, lineno=5)(_scope_1)
                    else:
                        {Placeholder.new('pack_1')}
                        return lambda : _cont(_scope_1)

                def _loop_exit_fn_1(_scope_1):
                    {Placeholder.new('unpack_1')}
                    x = x + 1
                    {Placeholder.new('pack_0')}
                    return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
                {Placeholder.new('pack_1')}
                return lambda : _cps.interpret(_loop_fn_1, cont=_loop_exit_fn_1, stack=_stack, func_src=__func_src, locals_=_locals_1, lineno=5)(_scope_1)
            else:
                {Placeholder.new('pack_0')}
                return lambda : _cont(_scope_0)

        def _loop_exit_fn_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont((rv, y))
        {Placeholder.new('pack_0')}
        return lambda : _cps.interpret(_loop_fn_0, cont=_loop_exit_fn_0, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
    ''', dict(
        **ScopeTools.pack_unpack(0, names, hashabledict=True),
        **ScopeTools.pack_unpack(1, names, hashabledict=True),
    ))
    check_cps_transform('''
    def fn(lim):
        rv = []
        x = 0
        while x < lim:
            y = 0
            while y < x:
                rv += [(x, y)]
                y = y + 1
            x = x + 1
        return (rv, y)
    ''', exp, check_args=[(3,)], check_out=[
        ([(1, 0), (2, 0), (2, 1)], 2),
    ])


def test_cps_while_call_in_test():
    # This test case is checking whether a call in the while loop's test is dynamically computed.
    # In the past, we could compile that call out when desugaring, which would mean any function
    # calls would be extracted into assignments to temporary vars. Our approach was to desugar
    # while loops to have a trivial test, which works well with our desugaring scheme.
    check_cps_transform('''
    def fn():
        ct = 0
        # To avoid control flow from `and`, we use &
        while (sum([ct]) < 3) & (ct < 100):
            ct += 1
        return ct
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fn(*, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        ct = 0

        @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
        def _loop_fn_0(_scope_0, *, _stack=(), _cps=_cps, _cont=lambda val: val):
            __func_src = "$IGNORE_STRING"
            {Placeholder.new('unpack_0')}
            if True:
                {Placeholder.new('pack_1')}

                def _cont_1(_res_1):
                    {Placeholder.new('unpack_1')}
                    def _cont_2(_scope_2):
                        {Placeholder.new('unpack_2')}
                        ct = ct + 1
                        {Placeholder.new('pack_0')}
                        return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=2)(_scope_0)
                    if not (_res_1 < 3) & (ct < 100):
                        {Placeholder.new('pack_0')}
                        return lambda : _cont(_scope_0)
                        {Placeholder.new('pack_2')}
                        return lambda : _cont_2(_scope_2)
                    else:
                        {Placeholder.new('pack_2')}
                        return lambda : _cont_2(_scope_2)
                return lambda : _cps.interpret(sum, cont=_cont_1, stack=_stack, func_src=__func_src, locals_=_locals_1, call_id=1, lineno=3)([ct])
            else:
                {Placeholder.new('pack_0')}
                return lambda : _cont(_scope_0)

        def _loop_exit_fn_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont(ct)
        {Placeholder.new('pack_0')}
        return lambda : _cps.interpret(_loop_fn_0, cont=_loop_exit_fn_0, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=2)(_scope_0)
    ''', {
        **ScopeTools.pack_unpack(0, ['ct'], hashabledict=True),
        **ScopeTools.pack_unpack(1, ['ct']),
        **ScopeTools.pack_unpack(2, ['ct']),
    }), check_args=[()], desugar=True)


def test_cps_while_nested_call():
    check_cps_transform('''
    def fn():
        x = 0
        ct = 0
        while ct < 10:
            x = sum([x, 1])
            x = sum([x, 2])
            ct = ct + 1
        return x
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, 'def fn():\\n    x = 0\\n    ct = 0\\n    while ct < 10:\\n        x = sum([x, 1])\\n        x = sum([x, 2])\\n        ct = ct + 1\\n    return x')
    def fn(*, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = 'def fn():\\n    x = 0\\n    ct = 0\\n    while ct < 10:\\n        x = sum([x, 1])\\n        x = sum([x, 2])\\n        ct = ct + 1\\n    return x'
        x = 0
        ct = 0

        @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
        def _loop_fn_0(_scope_0, *, _stack=(), _cps=_cps, _cont=lambda val: val):
            __func_src = "$IGNORE_STRING"
            {Placeholder.new('unpack_0')}
            if ct < 10:
                {Placeholder.new('pack_1')}
                def _cont_1(_res_1):
                    {Placeholder.new('unpack_1')}
                    x = _res_1
                    {Placeholder.new('pack_2')}
                    def _cont_2(_res_2):
                        {Placeholder.new('unpack_2')}
                        x = _res_2
                        ct = ct + 1
                        {Placeholder.new('pack_0')}
                        return lambda : _cps.interpret(_loop_fn_0, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
                    return lambda : _cps.interpret(sum, cont=_cont_2, stack=_stack, func_src=__func_src, locals_=_locals_2, call_id=2, lineno=5)([x, 2])
                return lambda : _cps.interpret(sum, cont=_cont_1, stack=_stack, func_src=__func_src, locals_=_locals_1, call_id=1, lineno=4)([x, 1])
            else:
                {Placeholder.new('pack_0')}
                return lambda : _cont(_scope_0)

        def _loop_exit_fn_0(_scope_0):
            {Placeholder.new('unpack_0')}
            return lambda : _cont(x)
        {Placeholder.new('pack_0')}
        return lambda : _cps.interpret(_loop_fn_0, cont=_loop_exit_fn_0, stack=_stack, func_src=__func_src, locals_=_locals_0, lineno=3)(_scope_0)
    ''', dict(
        **ScopeTools.pack_unpack(0, ['ct', 'x'], hashabledict=True),
        **ScopeTools.pack_unpack(1, ['ct', 'x']),
        **ScopeTools.pack_unpack(2, ['ct', 'x']),
    )), check_args=[()])


def test_cps_for():
    check_cps_transform('''
    def fn():
        rv = []
        for x in list(range(10)):
            rv += [x**2]
        return rv
    ''', ScopeTools.fill(f'''
    @lambda fn: CPSFunction(fn, '$FN_SOURCE')
    def fn(*, _stack=(), _cps=_cps, _cont=lambda val: val):
        __func_src = '$FN_SOURCE'
        rv = []
        {Placeholder.new('pack_1')}
        def _cont_1(_res_1):
            {Placeholder.new('unpack_1')}
            {Placeholder.new('pack_0')}
            def _cont_0(_res_0):
                {Placeholder.new('unpack_0')}
                _for_iter_4 = _res_0
                _for_idx_4 = 0

                @lambda fn: CPSFunction(fn, "$IGNORE_STRING")
                def _loop_fn_2(_scope_2, *, _stack=(), _cps=_cps, _cont=lambda val: val):
                    __func_src = "$IGNORE_STRING"
                    {Placeholder.new('unpack_2')}
                    if True:
                        {Placeholder.new('pack_3')}
                        def _cont_3(_res_3):
                            {Placeholder.new('unpack_3')}

                            def _cont_4(_scope_4):
                                {Placeholder.new('unpack_4')}
                                x = _for_iter_4[_for_idx_4]
                                _for_idx_4 = _for_idx_4 + 1
                                rv = rv + [x ** 2]
                                {Placeholder.new('pack_2')}
                                return lambda : _cps.interpret(_loop_fn_2, cont=_cont, stack=_stack, func_src=__func_src, locals_=_locals_2, lineno=4)(_scope_2)
                            if not _for_idx_4 < _res_3:
                                {Placeholder.new('pack_2')}
                                return lambda : _cont(_scope_2)
                                {Placeholder.new('pack_4')}
                                return lambda : _cont_4(_scope_4)
                            else:
                                {Placeholder.new('pack_4')}
                                return lambda : _cont_4(_scope_4)
                        return lambda : _cps.interpret(len, cont=_cont_3, stack=_stack, func_src=__func_src, locals_=_locals_3, call_id=3, lineno=5)(_for_iter_4)
                    else:
                        {Placeholder.new('pack_2')}
                        return lambda : _cont(_scope_2)

                def _loop_exit_fn_2(_scope_2):
                    {Placeholder.new('unpack_2')}
                    return lambda : _cont(rv)
                {Placeholder.new('pack_2')}
                return lambda : _cps.interpret(_loop_fn_2, cont=_loop_exit_fn_2, stack=_stack, func_src=__func_src, locals_=_locals_2, lineno=4)(_scope_2)
            return lambda : _cps.interpret(list, cont=_cont_0, stack=_stack, func_src=__func_src, locals_=_locals_0, call_id=0, lineno=2)(_res_1)
        return lambda : _cps.interpret(range, cont=_cont_1, stack=_stack, func_src=__func_src, locals_=_locals_1, call_id=1, lineno=2)(10)
    ''', {
        **ScopeTools.pack_unpack(0, ['rv']),
        **ScopeTools.pack_unpack(1, ['rv']),
        **ScopeTools.pack_unpack(2, ['_for_idx_4', '_for_iter_4', 'rv', 'x'], hashabledict=True),
        **ScopeTools.pack_unpack(3, ['_for_idx_4', '_for_iter_4', 'rv']),
        **ScopeTools.pack_unpack(4, ['_for_idx_4', '_for_iter_4', 'rv']),
    }), check_args=[()], desugar=True)


def _code_diff(a, b):
    from difflib import ndiff
    def _process_string(s):
        return (s.strip() + '\n').splitlines(keepends=True) # Make sure we have a newline at the very end
    diff = ndiff(_process_string(a), _process_string(b))
    return ''.join(diff)


def check_cps_transform(src, exp_src, *, check_args=[], check_out=None, desugar=False):
    src = textwrap.dedent(src)
    node = ast.parse(src)
    if desugar:
        node = DesugaringTransform()(node)
    # Replace $FN_SOURCE before we transform
    exp_src = exp_src.replace('$FN_SOURCE', ast.unparse(node).replace('\n', '\\n'))
    node = CPSTransform()(node)

    exp_src = textwrap.dedent(exp_src)
    exp_node = ast.parse(exp_src)

    assert_compare_ast(node, exp_node)

    assert (
        isinstance(node, ast.Module) and
        len(node.body) == 1 and
        isinstance(node.body[0], ast.FunctionDef)
    )
    fn_name = node.body[0].name

    src_context = {'CPSFunction': CPSFunction}
    exec(src, src_context)

    exp_context = {'CPSFunction': CPSFunction}
    interpreter = CPSInterpreter()
    exp_context[CPSTransform.cps_interpreter_name] = interpreter
    exec(exp_src, exp_context)

    assert check_args, 'Must supply test case to test transformed function.'
    if check_out is not None:
        assert len(check_out) == len(check_args)
    for idx, args in enumerate(check_args):
        expected = src_context[fn_name](*args)
        if check_out is not None:
            assert expected == check_out[idx]
        # We execute using only a simple trampoline, so this implementation can't handle stochastic primitives.
        assert expected == trampoline(exp_context[fn_name](*args))

def assert_compare_ast(node, exp_node):
    '''
    Helper function with nice error message for debugging.
    '''
    assert compare_ast(node, exp_node), f'Diff from expected source to transformed source.\n{_code_diff(ast.unparse(exp_node), ast.unparse(node))}\n\nTransformed source:\n{ast.unparse(node)}'

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
        # A special case to avoid including function sources.
        if isinstance(node1, str) and '$IGNORE_STRING' in (node1, node2):
            return True
        return node1 == node2

def test_Placeholder():
    trans = Placeholder.fill(ast.parse(textwrap.dedent(f'''
        x = {Placeholder.new('expr')}
        {Placeholder.new('stmt')}
        {Placeholder.new('stmts')}
    ''')), dict(
        expr=ast.Constant(3),
        stmt=ast.parse('y = f()').body[0],
        stmts=ast.parse('z=4\na=5').body,
    ))
    expected = ast.parse(textwrap.dedent('''
        x = 3
        y = f()
        z = 4
        a = 5
    '''))
    assert_compare_ast(trans, expected)
