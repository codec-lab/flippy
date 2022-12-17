import ast
import inspect
import textwrap
from typing import Tuple
from gorgo.core import ReturnState, SampleState, ObserveState, InitialState
from gorgo.core import StochasticPrimitive, ObservationStatement
from gorgo.transforms import DesugaringTransform, \
    CallWrap_and_Arg_Transform, SetLineNumbers, CPSTransform
from gorgo.funcutils import method_cache
import linecache
import builtins

class CPSInterpreter:
    def __init__(self):
        self.desugaring_transform = DesugaringTransform()
        self.call_transform = CallWrap_and_Arg_Transform()
        self.setlines_transform = SetLineNumbers()
        self.cps_transform = CPSTransform()

    def initial_program_state(self, function):
        interpreted_function = self.interpret(function)
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

    @method_cache
    def interpret(
        self,
        call,
        stack : Tuple = None, 
        func_src : str = None,
        locals_ = None,
        lineno : int = None,
    ):
        # normal python
        if isinstance(call, type):
            return self.interpret_class(call)
        if hasattr(call, '__name__') and getattr(builtins, call.__name__, None) == call:
            return self.interpret_builtin(call)

        # cps python
        if stack is None:
            cur_stack = None
        else:
            cur_stack = stack + ((func_src, lineno),)

        if CPSTransform.is_transformed(call):
            return self.interpret_transformed(call, cur_stack=cur_stack)
        if hasattr(call, "__self__"):
            if isinstance(call.__self__, StochasticPrimitive) and call.__name__ == "sample":
                return self.interpret_sample(call, cur_stack=cur_stack)
        if isinstance(call, ObservationStatement):
            return self.interpret_observation(call, cur_stack=cur_stack)
        return self.interpret_generic(call, cur_stack=cur_stack)

    def interpret_builtin(self, func):
        def builtin_wrapper(*args, _cont=lambda val: val, **kws):
            return _cont(func(*args, **kws))
        return builtin_wrapper

    def interpret_transformed(self, func, cur_stack):
        def wrapper_generic(*args, **kws):
            return func(*args, **kws, _cps=self, _stack=cur_stack)
        return wrapper_generic

    def interpret_sample(self, call, cur_stack):
        def sample_wrapper(_cont):
            return SampleState(
                continuation=_cont,
                distribution=call.__self__
            )
        return sample_wrapper

    def interpret_observation(self, func, cur_stack):
        def observation_wrapper(distribution, value, _cont=None, **kws):
            return ObserveState(
                continuation=lambda : _cont(None),
                distribution=distribution,
                value=value
            )
        return observation_wrapper

    def interpret_class(self, cls):
        def class_wrapper(*args, _cont=None, **kws):
            return lambda :  _cont(cls(*args, **kws))
        return class_wrapper

    def transform_from_func(self, func):
        trans_node = ast.parse(textwrap.dedent(inspect.getsource(func)))
        return self.transform(trans_node)

    def transform(self, trans_node):
        trans_node = self.desugaring_transform(trans_node)
        trans_node = self.call_transform(trans_node)
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

    def interpret_generic(self, func, cur_stack):
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
        def wrapper_generic(*args, **kws):
            return trans_func(*args, **kws, _cps=self, _stack=cur_stack)
        return wrapper_generic

    def get_closure(self, func):
        if getattr(func, "__closure__", None) is not None:
            closure_keys = func.__code__.co_freevars
            closure_values = [cell.cell_contents for cell in func.__closure__]
            return dict(zip(closure_keys, closure_values))
        else:
            return {}
