import ast
import functools
import inspect
import textwrap
import types
from collections import namedtuple
from typing import Union, TYPE_CHECKING
from gorgo.core import ReturnState, SampleState, ObserveState, InitialState
from gorgo.distributions.base import Distribution, Element
from gorgo.transforms import DesugaringTransform, \
    SetLineNumbers, CPSTransform, PythonSubsetValidator, ClosureScopeAnalysis
from gorgo.core import GlobalStore, ReadOnlyProxy
from gorgo.funcutils import method_cache
import linecache
import types
import contextlib

from gorgo.types import NonCPSCallable, Method, Continuation, Stack, \
    SampleCallable, ObserveCallable, CPSCallable, VariableName

StackFrame = namedtuple("StackFrame", "func_src lineno locals")

class CPSInterpreter:
    def __init__(self):
        self.subset_validator = PythonSubsetValidator()
        self.desugaring_transform = DesugaringTransform()
        self.closure_scope_analysis = ClosureScopeAnalysis()
        self.setlines_transform = SetLineNumbers()
        self.cps_transform = CPSTransform()
        self.global_store_proxy = ReadOnlyProxy()

    def initial_program_state(self, call: Union['NonCPSCallable','CPSCallable']) -> InitialState:
        cps_call = self.interpret(
            call=call,
            stack = ()
        )
        def return_continuation(value):
            return ReturnState(
                value=value
            )
        def program_continuation(*args, **kws):
            return cps_call(
                *args,
                _cont=return_continuation,
                **kws
            )
        return InitialState(
            continuation=program_continuation,
            cps=self
        )

    def interpret(
        self,
        call : Union['NonCPSCallable', 'CPSCallable'],
        cont : 'Continuation' = None,
        stack : 'Stack' = None,
        func_src : str = None,
        locals_ : dict = None,
        lineno : int = None,
    ) -> 'Continuation':
        """
        This is the main entry point for interpreting CPS-transformed code.
        See `CPSTransform.visit_Call` in `transforms.py` for more details on
        how it appears in transformed code.
        """

        # normal python
        if (
            isinstance(call, types.BuiltinFunctionType) or \
            isinstance(call, type) or
            (hasattr(call, "__self__") and isinstance(call.__self__, GlobalStore))
        ):
            continuation = self.interpret_builtin(call)
            return functools.partial(continuation, _cont=cont)

        # cps python
        continuation = self.interpret_cps(call)
        cur_stack = self.update_stack(stack, func_src, locals_, lineno)
        continuation = functools.partial(continuation, _stack=cur_stack, _cont=cont)
        return continuation

    def interpret_builtin(self, call: 'NonCPSCallable') -> 'Continuation':
        def builtin_continuation(*args, _cont: 'Continuation'=lambda val: val, **kws):
            return lambda : _cont(call(*args, **kws))
        return builtin_continuation

    def update_stack(
        self,
        stack: 'Stack',
        func_src: str,
        locals_: dict,
        lineno: int
    ) -> Union[None,'Stack']:
        if stack is None:
            cur_stack = None
        else:
            cur_stack = stack + (StackFrame(func_src, lineno, locals_),)
        return cur_stack

    @method_cache
    def interpret_cps(
        self,
        call : Union['NonCPSCallable', 'CPSCallable']
    ) -> 'Continuation':
        if CPSTransform.is_transformed(call):
            return self.interpret_transformed(call)
        if hasattr(call, "__self__") and isinstance(call.__self__, Distribution):
            if call.__name__ == "sample":
                return self.interpret_sample(call)
            elif call.__name__ == "observe":
                return self.interpret_observe(call)
            else:
                # other than sample and observe, we interpret Distribution methods as deterministic
                return self.interpret_method_deterministically(call)
        elif hasattr(call, "__self__"):
            raise NotImplementedError(f"CPSInterpreter does not support methods for {call.__self__.__class__.__name__}")
        return self.interpret_generic(call)

    def interpret_transformed(self, call : 'CPSCallable') -> 'Continuation':
        def generic_continuation(*args, _cont: 'Continuation'=None, _stack: 'Stack'=None, **kws):
            return call(*args, **kws, _cps=self, _stack=_stack, _cont=_cont)
        return generic_continuation

    def interpret_sample(self, call: 'SampleCallable[Element]') -> 'Continuation':
        def sample_continuation(
            _cont: 'Continuation'=None,
            _stack: 'Stack'=None,
            name: 'VariableName'=None
        ):
            return SampleState(
                continuation=_cont,
                distribution=call.__self__,
                name=name,
                stack=_stack,
                cps=self
            )
        return sample_continuation

    def interpret_observe(self, call: 'ObserveCallable[Element]') -> 'Continuation':
        def observe_continuation(
                value: 'Element',
                _cont: 'Continuation'=None,
                _stack: 'Stack'=None,
                name: 'VariableName'=None,
                **kws
            ):
            return ObserveState(
                continuation=lambda : _cont(None),
                distribution=call.__self__,
                value=value,
                name=name,
                stack=_stack,
                cps=self,
            )
        return observe_continuation

    def interpret_method_deterministically(self, call: Method) -> 'Continuation':
        self = call.__self__
        def method_continuation(*args, _cont: 'Continuation'=lambda val: val, _stack=None, **kws):
            return lambda : _cont(call.__func__(self, *args, **kws))
        return method_continuation

    def interpret_generic(self, call: 'NonCPSCallable') -> 'Continuation':
        code = self.compile(
            f'{call.__name__}_{hex(id(call)).removeprefix("0x")}.py',
            self.transform_from_func(call),
        )
        context = {**call.__globals__, **self.get_closure(call), "_cps": self, "global_store": self.global_store_proxy}
        try:
            exec(code, context)
        except SyntaxError as err :
            raise err
        trans_func = context[call.__name__]
        def generic_continuation(*args, _cont: 'Continuation'=lambda v: v, _stack: 'Stack'=None, **kws):
            return trans_func(*args, **kws, _cps=self, _stack=_stack, _cont=_cont)
        return generic_continuation

    def transform_from_func(self, func: 'NonCPSCallable') -> ast.AST:
        source = textwrap.dedent(inspect.getsource(func))
        trans_node = ast.parse(source)
        self.subset_validator(trans_node, source)
        return self.transform(trans_node)

    def transform(self, trans_node: ast.AST) -> ast.AST:
        self.closure_scope_analysis(trans_node)
        trans_node = self.desugaring_transform(trans_node)
        trans_node = self.setlines_transform(trans_node)
        trans_node = self.cps_transform(trans_node)
        return trans_node

    def compile(self, filename: str, node: ast.AST) -> types.CodeType:
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

    def get_closure(self, func: 'NonCPSCallable') -> dict:
        if getattr(func, "__closure__", None) is not None:
            closure_keys = func.__code__.co_freevars
            closure_values = [cell.cell_contents for cell in func.__closure__]
            return dict(zip(closure_keys, closure_values))
        else:
            return {}

    @contextlib.contextmanager
    def set_global_store(self, store : GlobalStore):
        assert self.global_store_proxy.proxied is None, 'Nested update of global store not supported.'
        try:
            self.global_store_proxy.proxied = store
            yield
        finally:
            self.global_store_proxy.proxied = None
