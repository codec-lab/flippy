import ast
import inspect
import textwrap
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
    def interpret(self, call):
        if CPSTransform.is_transformed(call):
            return self.interpret_transformed(call)
        if isinstance(call, type):
            return self.interpret_class(call)
        if hasattr(call, "__self__"):
            if isinstance(call.__self__, StochasticPrimitive) and call.__name__ == "sample":
                return self.interpret_sample(call)
        if isinstance(call, ObservationStatement):
            return self.interpret_observation(call)
        if getattr(builtins, call.__name__, None) == call:
            return self.interpret_builtin(call)
        return self.interpret_generic(call)

    def interpret_builtin(self, func):
        def builtin_wrapper(*args, _address=(), _cps=None, _cont=lambda val: val, **kws):
            return _cont(func(*args, **kws))
        return builtin_wrapper

    def interpret_transformed(self, func):
        def wrapper_generic(*args, _address=(), **kws):
            return func(*args, **kws, _cps=self, _address=_address)
        return wrapper_generic

    def interpret_sample(self, call):
        def sample_wrapper(_address, _cont):
            return SampleState(
                continuation=_cont,
                distribution=call.__self__
            )
        return sample_wrapper

    def interpret_observation(self, func):
        def observation_wrapper(*args, _address=None, _cont=None, **kws):
            return ObserveState(
                continuation=lambda : _cont(None),
                distribution=args[0] if len(args) >= 1 else kws['distribution'],
                value=args[1] if len(args) >= 2 else kws['value']
            )
        return observation_wrapper

    def interpret_class(self, cls):
        def class_wrapper(*args, _address=None, _cont=None, **kws):
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
        def wrapper_generic(*args, _address=(), **kws):
            return trans_func(*args, **kws, _cps=self, _address=_address)
        return wrapper_generic

    def get_closure(self, func):
        if getattr(func, "__closure__", None) is not None:
            closure_keys = func.__code__.co_freevars
            closure_values = [cell.cell_contents for cell in func.__closure__]
            return dict(zip(closure_keys, closure_values))
        else:
            return {}
