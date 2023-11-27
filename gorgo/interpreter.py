import ast
import functools
import inspect
import textwrap
import types
import dataclasses
from collections import namedtuple
from typing import Union, TYPE_CHECKING
from gorgo.core import ReturnState, SampleState, ObserveState, InitialState
from gorgo.distributions.base import Distribution, Element
from gorgo.transforms import DesugaringTransform, \
    SetLineNumbers, CPSTransform, PythonSubsetValidator, ClosureScopeAnalysis, \
    GetLineNumber
from gorgo.core import GlobalStore, ReadOnlyProxy
from gorgo.funcutils import method_cache
from gorgo.hashable import hashabledict
import linecache
import types
import contextlib

from gorgo.types import NonCPSCallable, Method, Continuation, Stack, \
    SampleCallable, ObserveCallable, CPSCallable, VariableName

@dataclasses.dataclass(frozen=True)
class StackFrame:
    func_src: str
    lineno: int
    locals: dict

    def as_string(self):
        func_string, line_match = self._func_src_string_line_match()
        func_string = ['   '+r for r in func_string]
        func_string[line_match] = func_string[line_match].replace('  ', '>>', 1)
        func_string[line_match] = func_string[line_match] + f'  # {self.locals}'
        return '\n'.join(func_string)

    def _func_src_string_line_match(self):
        try:
            func_ast = ast.parse(self.func_src).body[0]
            line = GetLineNumber()(func_ast, self.lineno)
            func_string = ast.unparse(func_ast).split('\n')
            line_string = ast.unparse(line)
        except SyntaxError:
            func_string = self.func_src.split('\n')
            line_string = func_string[self.lineno]
        line_match = [i for i, l in enumerate(func_string) if line_string in l]
        assert len(line_match) == 1
        line_match = line_match[0]
        return func_string, line_match

    def _repr_html_(self):
        func_string, line_match = self._func_src_string_line_match()
        func_string[line_match] = '<span style="color:red;">'+func_string[line_match]+'</span>'
        func_html = '<pre>'+'\n'.join(func_string)+'</pre>'
        func_html = func_html.replace('  ', '&nbsp;&nbsp;')
        locals_html = '<pre>'+'\n'.join([f'{k}: {v}' for k, v in self.locals.items()])+'</pre>'
        locals_keys = "<pre style='display:inline;'>"+', '.join(self.locals.keys())+'</pre>'
        func_head = "<pre style='display:inline;'>"+func_string[0].replace(":", "").replace("def ", "")+"</pre>"
        frame_html = [
            f"<details><summary>Locals: {locals_keys}</summary>{locals_html}</details>",
            f"<details><summary>Function: {func_head} </summary>{func_html}</details>"
        ]
        frame_html = "<div style='cursor:default'>"+'\n'.join(frame_html)+"</div>"
        return frame_html
        # return f'<pre>{self.as_string()}</pre>'

@dataclasses.dataclass(frozen=True)
class Stack:
    stack_frames: tuple = dataclasses.field(default_factory=tuple)

    def update(
        self,
        func_src: str,
        locals_: dict,
        lineno: int
    ) -> 'Stack':
        if isinstance(locals_, dict):
            locals_ = hashabledict({
                k: v for k, v in locals_.items()
                if (
                    (k not in ['__func_src', '_cont', '_cps', '_stack']) and
                    ('_scope_' not in k)
                )
            })
        new_stack = self.stack_frames + (StackFrame(func_src, lineno, locals_),)
        return Stack(new_stack)

    def as_string(self):
        return '\n'.join([f'Frame {i}:\n{frame.as_string()}\n' for i, frame in enumerate(self.stack_frames)])

    def __getitem__(self, key):
        return self.stack_frames[key]

    def _repr_html_(self):
        stack_html = []
        for i, frame in enumerate(self.stack_frames):
            frame_html = frame._repr_html_()
            frame_html = "<div style='margin-left: 20px;'>"+frame_html+"</div>"
            frame_html = f"<details open><summary>Frame {i}</summary>{frame_html}</details>"
            stack_html.append(frame_html)
        return '\n'.join(stack_html)


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
            stack = Stack(),
            func_src = "<root>",
            locals_ = {},
            lineno= 0
        )
        def return_continuation(value):
            return ReturnState(
                value=value,
                stack=Stack((StackFrame("<root>", 0, hashabledict({'__return__': value})), )),
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
        stack : Stack = Stack(),
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
        if not isinstance(stack, Stack):
            stack = Stack(stack)
        cur_stack = stack.update(func_src, locals_, lineno)
        continuation = functools.partial(continuation, _stack=cur_stack, _cont=cont)
        return continuation

    def interpret_builtin(self, call: 'NonCPSCallable') -> 'Continuation':
        def builtin_continuation(*args, _cont: 'Continuation'=lambda val: val, **kws):
            return lambda : _cont(call(*args, **kws))
        return builtin_continuation

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
