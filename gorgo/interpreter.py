import ast
import functools
import inspect
import textwrap
from typing import Tuple, Callable
from gorgo.core import ReturnState, SampleState, ObserveState, InitialState
from gorgo.core import StochasticPrimitive, ObservationStatement, StackFrame
from gorgo.transforms import DesugaringTransform, \
    SetLineNumbers, CPSTransform, PythonSubsetValidator
from gorgo.funcutils import method_cache
import linecache
import types

class CPSInterpreter:
    def __init__(self):
        self.subset_validator = PythonSubsetValidator()
        self.desugaring_transform = DesugaringTransform()
        self.setlines_transform = SetLineNumbers()
        self.cps_transform = CPSTransform()

    def initial_program_state(self, function):
        interpreted_function = self.interpret(
            function,
            stack = ()
        )
        def return_continuation(value):
            return ReturnState(
                value=value
            )
        def program_continuation(*args, **kws):
            return interpreted_function(
                *args,
                _cont=return_continuation,
                **kws
            )
        return InitialState(
            continuation=program_continuation,
        )

    def interpret(
        self,
        call,
        cont : Callable = None, 
        stack : Tuple = None, 
        func_src : str = None,
        locals_ = None,
        lineno : int = None,
    ):
        # normal python
        if (
            isinstance(call, types.BuiltinFunctionType) or \
            isinstance(call, type) 
        ):
            cps_call = self.interpret_builtin(call)
            return functools.partial(cps_call, _cont=cont)

        # cps python
        cps_call = self.interpret_cps(call)
        cur_stack = self.update_stack(stack, func_src, locals_, lineno)
        cps_call = functools.partial(cps_call, _stack=cur_stack, _cont=cont)
        return cps_call

    def interpret_builtin(self, func):
        def builtin_wrapper(*args, _cont=lambda val: val, **kws):
            return lambda : _cont(func(*args, **kws))
        return builtin_wrapper

    def update_stack(self, stack, func_src, locals_, lineno):
        if stack is None:
            cur_stack = None
        else:
            cur_stack = stack + (StackFrame(func_src, lineno, locals_),)
        return cur_stack

    @method_cache
    def interpret_cps(self, call):
        if CPSTransform.is_transformed(call):
            return self.interpret_transformed(call)
        if hasattr(call, "__self__"):
            if isinstance(call.__self__, StochasticPrimitive) and call.__name__ == "sample":
                return self.interpret_sample(call)
        if isinstance(call, ObservationStatement):
            return self.interpret_observation(call)
        return self.interpret_generic(call)

    def interpret_transformed(self, func):
        def wrapper_generic(*args, _cont=None, _stack=None, **kws):
            return func(*args, **kws, _cps=self, _stack=_stack, _cont=_cont)
        return wrapper_generic

    def interpret_sample(self, call):
        def sample_wrapper(_cont=None, _stack=None, name=None):
            return SampleState(
                continuation=_cont,
                distribution=call.__self__,
                name=name,
                stack=_stack,
            )
        return sample_wrapper

    def interpret_observation(self, call):
        def observation_wrapper(distribution, value, _cont=None, _stack=None, name=None, **kws):
            return ObserveState(
                continuation=lambda : _cont(None),
                distribution=distribution,
                value=value,
                name=name,
                stack=_stack
            )
        return observation_wrapper

    def interpret_generic(self, func):
        code = self.compile(
            f'{func.__name__}_{hex(id(func)).removeprefix("0x")}.py',
            self.transform_from_func(func),
        )
        local_context = {**self.get_closure(func), "_cps": self}
        try:
            exec(code, func.__globals__, local_context)
        except SyntaxError as err :
            raise err
        trans_func = local_context[func.__name__]
        def wrapper_generic(*args, _cont=None, _stack=None, **kws):
            return trans_func(*args, **kws, _cps=self, _stack=_stack, _cont=_cont)
        return wrapper_generic

    def transform_from_func(self, func):
        source = textwrap.dedent(inspect.getsource(func))
        trans_node = ast.parse(source)
        self.subset_validator(trans_node, source)
        return self.transform(trans_node)

    def transform(self, trans_node):
        trans_node = self.desugaring_transform(trans_node)
        trans_node = self.setlines_transform(trans_node)
        trans_node = self.cps_transform(trans_node)
        return trans_node

    def compile(self, filename, node):
        source = ast.unparse(node)
        # In order to get stack traces that reference compiled code, we follow the scheme IPython does
        # in CachingCompiler.cache, by adding an entry to Python's linecache.
        # https://github.com/ipython/ipython/blob/47abb68a/IPython/core/compilerop.py#L134-L178
        linecache.cache[filename] = (
            len(source),
            None,
            [line + "\n" for line in source.splitlines()],
            filename,
        )
        return compile(source, filename, 'exec')

    def get_closure(self, func):
        if getattr(func, "__closure__", None) is not None:
            closure_keys = func.__code__.co_freevars
            closure_values = [cell.cell_contents for cell in func.__closure__]
            return dict(zip(closure_keys, closure_values))
        else:
            return {}
