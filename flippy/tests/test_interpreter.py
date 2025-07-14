from flippy import recursive_map, recursive_filter, recursive_reduce, mem, flip
from flippy.distributions import Bernoulli, Categorical
from flippy.core import GlobalStore, ReadOnlyProxy
from flippy.core import SampleState, ReturnState
from flippy.interpreter import CPSInterpreter
from flippy.transforms import CPSTransform
from flippy.transforms import CPSFunction
from flippy.callentryexit import EnterCallState, ExitCallState, register_call_entryexit
from flippy.hashable import hashablelist, hashabledict
import ast
import math
import pytest
import traceback

from flippy.tests.test_transforms import trampoline

def geometric(p):
    x = Bernoulli(p).sample()
    if x == 0:
        return 0
    return 1 + geometric(p)

def algebra():
    def flip():
        return Bernoulli(0.5).sample()
    def NUM():
        return Categorical(range(5)).sample()
    def OP():
        return Categorical(['+', '*']).sample()
    def EQ():
        if flip():
            return NUM()
        else:
            return (NUM(), OP(), EQ())
    return EQ()

def check_trace(func, trace, *, args=(), kwargs={}, return_value):
    ps = CPSInterpreter().initial_program_state(func)
    print(ast.unparse(CPSInterpreter().transform_from_func(func)))
    ps = ps.step(*args, **kwargs)

    for trace_idx, (dist, value) in enumerate(trace):
        assert isinstance(ps, SampleState), (f'{trace_idx=}', ps)
        assert ps.distribution.isclose(dist), (f'{trace_idx=}', ps)
        assert value in dist.support
        ps = ps.step(value)

    assert isinstance(ps, ReturnState)
    assert ps.value == return_value

def test_interpreter():
    check_trace(geometric, [
        (Bernoulli(0.9), 0),
    ], args=(0.9,), return_value=0)

    check_trace(geometric, [
        (Bernoulli(0.8), 1),
        (Bernoulli(0.8), 1),
        (Bernoulli(0.8), 0),
    ], args=(0.8,), return_value=2)

    check_trace(algebra, [
        (Bernoulli(0.5), 1),
        (Categorical(range(5)), 3),
    ], return_value=3)

    check_trace(algebra, [
        (Bernoulli(0.5), 0),
        (Categorical(range(5)), 3),
        (Categorical(['+', '*']), '*'),
        # subtree on right
        (Bernoulli(0.5), 0),
        (Categorical(range(5)), 2),
        (Categorical(['+', '*']), '+'),
        (Bernoulli(0.5), 1),
        (Categorical(range(5)), 4),
    ], return_value=(3, '*', (2, '+', 4)))

def test_cps_map():
    def fn():
        def f(x):
            return Bernoulli(x).sample()
        return sum(recursive_map(f, [.1, .2]))

    check_trace(fn, [
        (Bernoulli(0.1), 1),
        (Bernoulli(0.2), 1),
    ], return_value=2)

def test_cps_filter():
    def fn():
        def f(x):
            return Categorical([1, 2, 3, 4]).sample()
        def is_even(x):
            return x % 2 == 0
        return sum(recursive_filter(is_even, recursive_map(f, [None] * 4)))

    check_trace(fn, [
        (Categorical([1, 2, 3, 4]), 1),
        (Categorical([1, 2, 3, 4]), 2),
        (Categorical([1, 2, 3, 4]), 3),
        (Categorical([1, 2, 3, 4]), 4),
    ], return_value=6)

def test_recursive_reduce():
    def fn():
        return recursive_reduce(lambda acc, x: acc + Bernoulli(x).sample(), [.1, .2], 0)

    check_trace(fn, [
        (Bernoulli(0.1), 1),
        (Bernoulli(0.2), 1),
    ], return_value=2)

def test_list_comprehension():
    # Simple case
    expected = [0, 1, 4]
    def fn():
        return [x**2 for x in range(3)]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Multiple if statements.
    expected = [0, 2]
    def fn():
        return [x for x in range(5) if x % 2 == 0 if x < 3]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Multiple generators
    expected = [(0, 0), (0, 1), (0, 2), (2, 0)]
    def fn():
        return [
            (x, y)
            for x in range(4)
            if x % 2 == 0
            for y in range(5)
            if x + y < 3
        ]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Nested comprehensions
    expected = [[0], [0, 1], [0, 1, 2]]
    def fn():
        return [
            [y for y in range(x+1)]
            for x in range(3)
        ]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Set comprehensions
    expected = {0, 1, 4, 9}
    def fn():
        return {
            x**2
            for x in [-3, -2, -1, 0, 1, 2, 3]
        }
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Dict comprehensions
    expected = {0: 0, 1: 1, 2: 4, 3: 9}
    def fn():
        return {
            x: x**2
            for x in range(4)
        }
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Checking something stochastic.
    def fn():
        return sum([Bernoulli(x).sample() for x in [.1, .2, .3]])
    check_trace(fn, [
        (Bernoulli(0.1), 1),
        (Bernoulli(0.2), 0),
        (Bernoulli(0.3), 1),
    ], return_value=2)

def test_check_exception():
    def test_fn():
        raise Exception('expected exception')
        return 3
    with pytest.raises(Exception) as e:
        check_trace(test_fn, [], return_value=3)
    # The right exception was raised.
    assert 'expected exception' in str(e)

    # Now check exception content to make sure it references code.
    _, exc, tb = e._excinfo
    # We show compiled code, which in this case isn't particularly readable.
    # This line will have to change depending on our compiled output.
    exception_line = 'raise _res_0'

    # First, we just check that the traceback will render the code.
    formatted = ''.join(traceback.format_exception(exc, None, tb))
    assert exception_line in formatted

    # This is a more detailed check of the contents, ensuring the filename is correct in traceback.
    last_entry = traceback.extract_tb(tb)[-1]
    assert 'test_fn' in last_entry.filename
    assert hex(id(test_fn)).removeprefix('0x') in last_entry.filename
    assert last_entry.line == exception_line

def test_control_flow_or():
    def fn_or():
        return Bernoulli(0.5).sample() or Bernoulli(0.5).sample()

    check_trace(fn_or, [
        (Bernoulli(0.5), 0),
        (Bernoulli(0.5), 1),
    ], return_value=1)

    check_trace(fn_or, [
        (Bernoulli(0.5), 1),
    ], return_value=1)

def test_control_flow_and():
    def fn_and():
        return Bernoulli(0.5).sample() and Bernoulli(0.5).sample()

    check_trace(fn_and, [
        (Bernoulli(0.5), 1),
        (Bernoulli(0.5), 1),
    ], return_value=1)

    check_trace(fn_and, [
        (Bernoulli(0.5), 0),
    ], return_value=0)

def test_control_flow_for_continue_break():
    def fn():
        rv = []
        for i in range(10):
            if i == 3:
                continue
            if i == 7:
                break
            rv += [i]
        return rv

    check_trace(fn, [], return_value=[0, 1, 2, 4, 5, 6])

def test_conditional_reassign():
    def fn():
        x = 0
        if Bernoulli(0.5).sample():
            x = 42
        return x

    check_trace(fn, [(Bernoulli(0.5), 0)], return_value=0)
    check_trace(fn, [(Bernoulli(0.5), 1)], return_value=42)

def test_load_then_store_in_new_scope():
    def fn():
        x = 0
        z = Bernoulli(0.5).sample()
        x = x + z
        return x

    check_trace(fn, [(Bernoulli(0.5), 0)], return_value=0)
    check_trace(fn, [(Bernoulli(0.5), 1)], return_value=1)

def test_conditionally_defined():
    # This can break implementations of scope passing that assume
    # variable lists can be fully statically determined.
    def fn():
        z = Bernoulli(0.5).sample()

        if z:
            y = 42

        if z:
            return y
        else:
            return -1

    check_trace(fn, [(Bernoulli(0.5), 0)], return_value=-1)
    check_trace(fn, [(Bernoulli(0.5), 1)], return_value=42)

def test_closure_issues():
    # Making sure we throw for important closure issues.

    # If incorrect, this returns 'bad'.
    with pytest.raises(SyntaxError) as err:
        def fn():
            def nested():
                return x
            x = 'bad'
            Bernoulli(0.5).sample() # Arbitrary. To put rest of function in continuation.
            x = 'good'
            return nested()
        check_trace(fn, [(Bernoulli(0.5), 0)], return_value='good')
    assert 'must be immutable' in str(err)

    # If incorrect, this fails to run with NameError.
    with pytest.raises(SyntaxError) as err:
        def fn():
            def nested():
                return x
            Bernoulli(0.5).sample() # Arbitrary. To put rest of function in continuation.
            x = 'good'
            return nested()
        check_trace(fn, [(Bernoulli(0.5), 0)], return_value='good')
    assert 'must be defined before' in str(err)

def test_global_store_proxy():
    global_store = None
    def f():
        return global_store.get('a', 'no value')

    cps = CPSInterpreter()
    code = cps.transform_from_func(f)
    context = {
        "_cps": cps,
        'global_store': cps.global_store_proxy,
        "CPSFunction": CPSFunction,
        "hashablelist": hashablelist,
        "hashabledict": hashabledict,
    }
    exec(ast.unparse(code), context)
    f = context['f']

    with cps.set_global_store(GlobalStore()):
        assert trampoline(f()) == 'no value'

    with cps.set_global_store(GlobalStore({'a': 100})):
        assert trampoline(f()) == 100

proxy_forking_bags = Categorical(['bag0', 'bag1', 'bag2'])
def proxy_forking_value(_bag):
    return Categorical(range(10)).sample()
proxy_forking_value = CPSInterpreter().non_cps_callable_to_cps_callable(proxy_forking_value)
def proxy_forking():
    value = mem(proxy_forking_value)
    return [
        value(proxy_forking_bags.sample())
        for _ in range(5)
    ]

def test_global_store_proxy_forking():
    cps = CPSInterpreter()
    s0 = cps.initial_program_state(proxy_forking).step()
    assert s0.init_global_store.store == {}

    s1a = s0.step('bag0').step(3)
    assert s0.init_global_store.store == {}, 'Make sure original state was not modified'
    assert s1a.init_global_store.store == {(proxy_forking_value, ('bag0',), ()): 3}

    # A sanity check, we shouldn't need to resample for bag0 now.
    s = s1a.step('bag0')
    assert isinstance(s, SampleState) and s.distribution == proxy_forking_bags

    # Now, we test forking by restarting at s0
    s1b = s0.step('bag1').step(7)
    assert s0.init_global_store.store == {}, 'Make sure original state was not modified'
    assert s1a.init_global_store.store == {(proxy_forking_value, ('bag0',), ()): 3}, 'Make sure sibling state was not modified'
    assert s1b.init_global_store.store == {(proxy_forking_value, ('bag1',), ()): 7}

    # This forked state should not pick up on state from s1a
    s2b = s1b.step('bag0').step(5)
    assert s0.init_global_store.store == {}, 'Make sure original state was not modified'
    assert s1a.init_global_store.store == {(proxy_forking_value, ('bag0',), ()): 3}, 'Make sure sibling state was not modified'
    assert s1b.init_global_store.store == {(proxy_forking_value, ('bag1',), ()): 7}, 'Make sure original state was not modified'
    assert s2b.init_global_store.store == {(proxy_forking_value, ('bag1',), ()): 7, (proxy_forking_value, ('bag0',), ()): 5}, 'Make sure original state was not modified'

def test_Distribution_generic_methods():
    # other than observe and sample, Distribution methods
    # should be run deterministically
    def f():
        x = Bernoulli(0.5)
        return x.log_probability(1)
    check_trace(
        f, [], return_value=math.log(.5)
    )

def test_global_store_programstate_hashing():
    def g_base(i):
        return 100
    g_base = CPSInterpreter().non_cps_callable_to_cps_callable(g_base)
    g = mem(g_base)

    def f_global_diff():
        i = flip(.2, name='i')
        j = g(i)
        return j

    def f_global_same():
        i = flip(.2, name='i')
        j = g(0)
        return j

    def f_noglobal():
        i = flip(.2, name='i')
        j = g_base(i)
        return j

    ps = CPSInterpreter().initial_program_state(f_global_diff)
    i_ps = ps.step()
    assert i_ps.step(1) != i_ps.step(0)
    assert (g_base, (True, ), ()) in i_ps.step(1).init_global_store.store
    assert (g_base, (False, ), ()) in i_ps.step(0).init_global_store.store
    assert i_ps.step(1).value == i_ps.step(0).value == 100

    ps = CPSInterpreter().initial_program_state(f_global_same)
    i_ps = ps.step()
    assert i_ps.step(1) == i_ps.step(0)
    assert (g_base, (False, ), ()) in i_ps.step(0).init_global_store.store
    assert (g_base, (False, ), ()) in i_ps.step(1).init_global_store.store
    assert i_ps.step(1).value == i_ps.step(0).value == 100

    ps = CPSInterpreter().initial_program_state(f_noglobal)
    i_ps = ps.step()
    assert i_ps.step(1) == i_ps.step(0)
    assert i_ps.step(0).init_global_store.store == i_ps.step(1).init_global_store.store == {}
    assert i_ps.step(1).value == i_ps.step(0).value == 100

def test_deterministic_nested_call_entryexit():
    def model():
        @register_call_entryexit
        def f1():
            return 'a'

        @register_call_entryexit
        def f2():
            return 'b' + f1()

        @register_call_entryexit
        def f3():
            return 'c' + f2()
        return f3()
    ps = CPSInterpreter().initial_program_state(model)
    ps_seq = []
    while not isinstance(ps, ReturnState):
        ps = ps.step()
        ps_seq.append(ps)
    assert isinstance(ps_seq[0], EnterCallState)
    assert ps_seq[0].function.__name__ == 'f3'
    assert isinstance(ps_seq[1], EnterCallState)
    assert ps_seq[1].function.__name__ == 'f2'
    assert isinstance(ps_seq[2], EnterCallState)
    assert ps_seq[2].function.__name__ == 'f1'
    assert isinstance(ps_seq[3], ExitCallState)
    assert ps_seq[3].function.__name__ == 'f1'
    assert isinstance(ps_seq[4], ExitCallState)
    assert ps_seq[4].function.__name__ == 'f2'
    assert isinstance(ps_seq[5], ExitCallState)
    assert ps_seq[5].function.__name__ == 'f3'

    assert ps.value == 'cba'

def test_call_entry_exit_cps_function_requirement():
    with pytest.raises(AssertionError) as e:
        @register_call_entryexit
        def f1():
            return 'a'
    assert 'can only be applied to transformed functions' in str(e)

def test_program_state_identity_with_closures():
    def f():
        i = flip(.65, name='i')
        def g():
            j = flip(.23, name='j')
            return j + i
        return g()

    ps = CPSInterpreter().initial_program_state(f)
    ps = ps.step()

    # identity of program states depends on g() and its closure (which includes i)
    ps_0a = ps.step(0)
    ps_0b = ps.step(0)
    ps_1 = ps.step(1)
    assert ps_0a == ps_0b
    assert hash(ps_0a) == hash(ps_0b)
    assert id(ps_0a) != id(ps_0b)
    assert ps_0a != ps_1
    assert hash(ps_0a) != hash(ps_1)

    # Names are the same regardless of locals
    assert ps_0a.name == ps_0b.name
    assert ps_0a.name == ps_1.name

    # g in each thread are equal, but generally not identical objects
    assert ps_0a.stack[1].locals['g'] == ps_0b.stack[1].locals['g']
    assert id(ps_0a.stack[1].locals['g']) != id(ps_0b.stack[1].locals['g'])
    # try:
    #     # its possible but unlikely that they are identical due to memory reuse
    #     assert id(ps.step(0).stack[1].locals['g']) != id(ps.step(0).stack[1].locals['g'])
    # except AssertionError:
    #     assert id(ps.step(0).stack[1].locals['g']) != id(ps.step(0).stack[1].locals['g'])

def _assert_stack_frame_fn_match(stack, fn_names):
    assert len(stack.stack_frames) == len(fn_names)
    for frame, fn_name in zip(stack.stack_frames, fn_names):
        if fn_name == '<root>':
            prefix = fn_name
        else:
            prefix = f'def {fn_name}('
        assert frame.func_src.startswith(prefix), (prefix, frame.func_src[:100])

def test_program_state_identity_with_loops():
    def f():
        ct = 0
        for _ in range(3):
            ct += Bernoulli(0.5).sample()
        return ct

    ps = CPSInterpreter().initial_program_state(f)
    ps = ps.step()
    ps0 = ps.step(0)
    ps00 = ps0.step(0)
    assert ps0 != ps00
    assert hash(ps0) != hash(ps00)
    assert ps0.name != ps00.name
    _assert_stack_frame_fn_match(ps0.name, ['<root>', 'f', '_loop_fn_1', '_loop_fn_1'])
    _assert_stack_frame_fn_match(ps00.name, ['<root>', 'f', '_loop_fn_1', '_loop_fn_1', '_loop_fn_1'])

def test_program_state_identity_with_loop_continue():
    def f():
        ct = 0
        for i in range(3):
            if Bernoulli(0.5).sample():
                ct += i * 2
                continue
            ct += i
        return ct

    ps = CPSInterpreter().initial_program_state(f)
    ps = ps.step()
    ps0 = ps.step(0)
    ps00 = ps0.step(0)
    ps1 = ps.step(1)
    ps10 = ps1.step(0)
    assert ps00 != ps10
    assert hash(ps00) != hash(ps10)
    # TODO FIX THIS
    with pytest.raises(AssertionError):
        assert ps00.name != ps10.name
        # HACK: make this check for line numbers
        _assert_stack_frame_fn_match(ps0.name, ['<root>', 'f', '_loop_fn_5', '_loop_fn_5'])
        _assert_stack_frame_fn_match(ps00.name, ['<root>', 'f', '_loop_fn_5', '_loop_fn_5', '_loop_fn_5'])

def test_call_entryexit_skipping():
    def model():
        @register_call_entryexit
        def f(p):
            return flip(p) + flip(p)
        return f(.6) + 20

    ps = CPSInterpreter().initial_program_state(model)
    enter_ps = ps.step()

    # Don't run the model and return 100
    exit_ps = enter_ps.step(False, 100)
    assert exit_ps.value == 100
    return_ps = exit_ps.step()
    assert return_ps.value == 120

    # Run the function and set bernoulli's to 1
    exit_ps = enter_ps.step(True).step(1).step(1)
    assert exit_ps.value == 2
    return_ps = exit_ps.step()
    assert return_ps.value == 22

def test_decorated_recursive_functions():
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper

    def f():
        @decorator
        def wrapped():
            inner_ref = wrapped
            return inner_ref
        outer_ref = wrapped
        inner_ref = wrapped()
        assert outer_ref is inner_ref
        return 1

    ps = CPSInterpreter().initial_program_state(f)
    ps.step()

def test_program_state_stack_name_with_local_change():
    def f(i):
        x = Bernoulli(0.5).sample()
        return x

    ps = CPSInterpreter().initial_program_state(f)
    assert ps.step(0).name == ps.step(1).name

def test_sampling_in_instance_method_call():
    class C:
        def flip(self, p):
            return Bernoulli(p).sample()

    def call_as_instance_attribute():
        c = C()
        return c.flip(.63)

    def call_as_class_attribute():
        c = C()
        return C.flip(c, .63)

    def tests(f):
        # Execute to the sample statement in the class method
        in_class_flip_ps = CPSInterpreter().initial_program_state(f).step()
        assert isinstance(in_class_flip_ps, SampleState)
        assert in_class_flip_ps.distribution.isclose(Bernoulli(.63))

        # Check that instance is referenced in transformed method
        class_flip_locals = in_class_flip_ps.stack[-1].locals
        assert isinstance(class_flip_locals['self'], C)

        # Check instance properties in main function scope
        f_class_locals = in_class_flip_ps.stack[-2].locals
        assert class_flip_locals['self'] is f_class_locals['c']
        assert isinstance(f_class_locals['c'], C)
        assert f_class_locals['c'].flip.__qualname__[-len("C.flip"):] == "C.flip"

    tests(call_as_instance_attribute)
    tests(call_as_class_attribute)

def test_transformed_class_method_does_not_shadow():
    def func(p):
        return 100
    nonlocal_func = func

    # Test instance method
    class C:
        def func(self):
            assert func is nonlocal_func, "nonlocal function has been redefined"
            return 101

    def f():
        c = C()
        return c.func()

    assert f() == 101
    assert CPSInterpreter().initial_program_state(f).step().value == 101

    # Now test class method
    class C:
        @classmethod
        def func(cls):
            assert func is nonlocal_func, "nonlocal function has been redefined"
            return 101

    def f_class():
        return C.func()

    assert f_class() == 101
    assert CPSInterpreter().initial_program_state(f_class).step().value == 101

    # Now test static method
    class C:
        @staticmethod
        def func():
            assert func is nonlocal_func, "nonlocal function has been redefined"
            return 101

    def f_static():
        return C.func()

    assert f_static() == 101
    assert CPSInterpreter().initial_program_state(f_static).step().value == 101

    # Now test nested class method
    class C1:
        class C2:
            def func(self):
                assert func is nonlocal_func, "nonlocal function has been redefined"
                return 101

    def f_nested():
        c1 = C1()
        c2 = c1.C2()
        return c2.func()

    assert CPSInterpreter().initial_program_state(f_nested).step().value == 101

def test_calling_transformed_staticmethod():
    class C:
        @staticmethod
        def flip():
            p = .66
            return Bernoulli(p).sample()

        @staticmethod
        def flip2(p):
            return Bernoulli(p).sample()

        def flip_no_decorator():
            p = .66
            return Bernoulli(p).sample()

    def call_from_class():
        return C.flip()

    def call_from_class_with_arg():
        return C.flip2(.66)

    def call_from_class_no_decorator():
        return C.flip_no_decorator()

    def call_from_instance():
        c = C()
        return c.flip()

    def call_from_instance_no_decorator():
        c = C()
        return c.flip_no_decorator()

    ps = CPSInterpreter().initial_program_state(call_from_class).step()
    assert ps.distribution.isclose(Bernoulli(.66))

    ps = CPSInterpreter().initial_program_state(call_from_class_with_arg).step()
    assert ps.distribution.isclose(Bernoulli(.66))

    ps = CPSInterpreter().initial_program_state(call_from_class_no_decorator).step()
    assert ps.distribution.isclose(Bernoulli(.66))

    ps = CPSInterpreter().initial_program_state(call_from_instance).step()
    assert ps.distribution.isclose(Bernoulli(.66))

    with pytest.raises(TypeError) as e:
        ps = CPSInterpreter().initial_program_state(call_from_instance_no_decorator).step()
        assert isinstance(e.value, TypeError) and "0 positional argument" in str(e.value)

def test_calling_transformed_classmethod():
    class C:
        p = .66
        def __init__(self):
            self.p = .22
        @classmethod
        def flip(cls):
            return Bernoulli(cls.p).sample()

    def call_from_instance():
        c = C()
        return c.flip()

    def call_from_class():
        return C.flip()

    ps = CPSInterpreter().initial_program_state(call_from_instance).step()
    assert ps.distribution.isclose(Bernoulli(.66))

    ps = CPSInterpreter().initial_program_state(call_from_class).step()
    assert ps.distribution.isclose(Bernoulli(.66))

def test_generate_unique_method_name():
    def func(): pass

    class C:
        def func(self): pass
        class NC: #nested class
            def func(self): pass
    c = C()
    nc = C.NC()

    def class_factory():
        class LC: #locally defined class
            def func(self): pass
        return LC

    LC = class_factory()
    lc = LC()

    funcs_names = [
        (func, None),
        (C.func, '__C_func'),
        (c.func, '__C_func'),
        (C.NC.func, "__C_NC_func"),
        (nc.func, "__C_NC_func"),
        (LC.func, '__LC_func'),
        (lc.func, '__LC_func')
    ]
    for f, n in funcs_names:
        gen_name = CPSInterpreter().generate_unique_method_name(f)
        assert n == gen_name, (f, n, gen_name)

def test_CPSFunction_hashing_and_equality():
    # We assume functions are equal up to source code and closure
    # This doesn't include global scope since that's handled by program state
    def model():
        i = Bernoulli(0.5).sample() + Bernoulli(0.5).sample()
        def f():
            return i + 1
        return f

    ps = CPSInterpreter().initial_program_state(model)
    fs = [ps.step().step(b1).step(b2).value for b1 in [0, 1] for b2 in [0, 1]]
    assert len(set(fs)) == 3, "Functions are unique up to their source code and closures"
    assert len(set(f.func_src for f in fs)) == 1
    assert len(set(f.closure for f in fs)) == 3
    assert set(f.closure['i'] for f in fs) == {0, 1, 2}

def test_callsite_addressing_on_single_line():
    def m():
        def f(i):
            return 1
        return f(f(f(1)))

    ps = CPSInterpreter(_emit_call_entryexit=True).initial_program_state(m)
    ps = ps.step().step()
    call1 : EnterCallState = ps
    ps = ps.step().step()
    call2 : EnterCallState = ps
    ps = ps.step().step()
    call3 : EnterCallState = ps

    # Address components we expect to stay the same for different calls
    call_names = (call1.function.__name__, call2.function.__name__, call3.function.__name__)
    assert set(call_names) == {"f"}, "Call function names should be the same"
    call_linenos = (call1.stack[-1].lineno, call2.stack[-1].lineno, call3.stack[-1].lineno)
    assert len(set(call_linenos)) == 1, "Call line numbers should be the same"
    call_substacks = (call1.stack[:-1], call2.stack[:-1], call3.stack[:-1])
    assert len(set(call_substacks)) == 1, "Call sub stacks should be the same"

    # Address components we expect to be distinct for different calls
    call_ids = (call1.stack[-1].call_id, call2.stack[-1].call_id, call3.stack[-1].call_id)
    assert len(set(call_ids)) == 3, "Call IDs should be unique"

    # Thus the call stacks should be distinct
    call_stacks = (call1.stack, call2.stack, call3.stack)
    assert len(set(call_stacks)) == 3, "Call stacks should be the same"
