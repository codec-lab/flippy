"""
This moduule implements a continuation-passing style (CPS) interpreter for Python code,
which allows us to intercept sample and observe statements as described by
[van de Meent et al. (2018)](https://arxiv.org/abs/1809.10756).
"""

import ast
import functools
import inspect
import textwrap
import types
from types import MethodType
import dataclasses
import linecache
import contextlib
from typing import Union, TYPE_CHECKING, Tuple, Callable, TypeVar, Any

from flippy.core import ReturnState, SampleState, ObserveState, InitialState
from flippy.callentryexit import EnterCallState, ExitCallState
from flippy.distributions.base import Distribution, Element
from flippy.transforms import DesugaringTransform, \
    SetLineNumbers, CPSTransform, PythonSubsetValidator, ClosureScopeAnalysis, \
    GetLineNumber, CPSFunction, HashableCollectionTransform
from flippy.core import GlobalStore, ReadOnlyProxy
from flippy.funcutils import method_cache
from flippy.hashable import hashabledict, hashablelist

from flippy.types import NonCPSCallable, Method, Continuation, \
    SampleCallable, ObserveCallable, CPSCallable, VariableName

# Note if we use python 3.10+ we can use typing.ParamSpec
# so that combinators preserve the type signature of functions
R = TypeVar('R')

@dataclasses.dataclass(frozen=True)
class StackFrame:
    """
    @private
    The `StackFrame` class represents a single frame in the stack of function calls.
    It contains the source code of the function, the line number of the call,
    a unique call identifier, and the local variables at the time of the call.
    """
    func_src: str
    lineno: int
    call_id: int
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
        func_string = [r.replace('<', '&lt;').replace('>', '&gt;') for r in func_string]
        func_string[line_match] = '<span style="color:red;">'+func_string[line_match]+'</span>'
        func_html = '<pre>'+'\n'.join(func_string)+'</pre>'
        func_html = func_html.replace('  ', '&nbsp;&nbsp;')
        if self.locals is None:
            locals_html = '<pre>None</pre>'
            locals_keys = "<pre style='display:inline;'></pre>"
        else:
            locals_html = '<pre>'+'\n'.join([
                f'{k}: {v}'.replace('<', '&lt;').replace('>', '&gt;')
                for k, v in self.locals.items()
            ])+'</pre>'
            locals_keys = "<pre style='display:inline;'>"+', '.join(self.locals.keys())+'</pre>'
        func_head = "<pre style='display:inline;'>"+func_string[0].replace(":", "").replace("def ", "")+"</pre>"
        frame_html = [
            f"<details><summary>Locals: {locals_keys}</summary>{locals_html}</details>",
            f"<details><summary>Caller: {func_head} </summary>{func_html}</details>"
        ]
        frame_html = "<div style='cursor:default'>"+'\n'.join(frame_html)+"</div>"
        return frame_html

@dataclasses.dataclass(frozen=True)
class Stack:
    """
    @private
    The `Stack` class represents a stack of function call frames.
    Note that the stack is not actually used in execution since
    CPS-transformed programs are executed with a trampoline, but it provides
    a way to identify the current state of execution for debugging and
    addressing.
    """
    stack_frames: Tuple[StackFrame] = dataclasses.field(default_factory=tuple)

    def update(
        self,
        func_src: str,
        locals_: dict,
        call_id: int,
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
        new_stack = self.stack_frames + (StackFrame(func_src, lineno, call_id, locals_),)
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

    def without_locals(self):
        return Stack(tuple([
            StackFrame(f.func_src, f.lineno, f.call_id, None) for f in self.stack_frames
        ]))

    def __len__(self):
        return len(self.stack_frames)

class CPSInterpreter:
    """
    @private
    The `CPSInterpreter` class is responsible for
    first transforming Python functions into CPS form using
    source-to-source transformations found in `flippy.transforms` and then
    executing the transformed code. The `CPSInterpreter` also handles the
    the interpretation of various callable types, including normal Python functions,
    class methods, static methods, and distribution methods and provides
    a way to interact with program execution with `ProgramState` objects.
    """

    _compile_mode = False
    _decorator_var_name = "__decorator_vars__"
    def __init__(self, _emit_call_entryexit: bool = False):
        self.subset_validator = PythonSubsetValidator()
        self.desugaring_transform = DesugaringTransform()
        self.hashable_collection_transform = HashableCollectionTransform()
        self.closure_scope_analysis = ClosureScopeAnalysis()
        self.setlines_transform = SetLineNumbers()
        self.cps_transform = CPSTransform()
        self.global_store_proxy = ReadOnlyProxy()
        self._emit_call_entryexit = _emit_call_entryexit

    def initial_program_state(self, call: Union['NonCPSCallable','CPSCallable']) -> InitialState:
        cps_call = self.interpret(
            call=call,
            stack = Stack(),
            func_src = "<root>",
            locals_ = {},
            call_id=None,
            lineno= 0
        )
        def return_continuation(value):
            return ReturnState(
                value=value,
                stack=Stack((StackFrame("<root>", 0, None, hashabledict({'__return__': value})), )),
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
        call_id : int = None,
        lineno : int = None,
    ) -> 'Continuation':
        """
        This is the main entry point for interpreting CPS-transformed code.
        See `CPSTransform.visit_Call` in `transforms.py` for more details on
        how it appears in transformed code.

        `call_id` is used to uniquely identify function calls in the original code.
        `lineno` is used to track the line number of the call in the source
        code for debugging purposes.
        """

        # normal python
        if (
            isinstance(call, types.BuiltinFunctionType) or \
            isinstance(call, type) or
            (isinstance(call, MethodType) and isinstance(call.__self__, GlobalStore))
        ):
            continuation = self.interpret_builtin(call)
            return functools.partial(continuation, _cont=cont)

        # cps python
        if isinstance(call, MethodType):
            # Only instance/class methods (not static methods) follow this path
            continuation = self.interpret_object_attribute_call(call)
        else:
            continuation = self.interpret_cps(call)
        if not isinstance(stack, Stack):
            stack = Stack(stack)
        cur_stack = stack.update(func_src, locals_, call_id, lineno)
        continuation = functools.partial(continuation, _stack=cur_stack, _cont=cont)
        return continuation

    def interpret_builtin(self, call: 'NonCPSCallable') -> 'Continuation':
        def builtin_continuation(*args, _cont: 'Continuation'=lambda val: val, **kws):
            return lambda : _cont(call(*args, **kws))
        return builtin_continuation

    def interpret_object_attribute_call(self, call: 'NonCPSCallable') -> 'Continuation':
        if isinstance(call.__self__, Distribution):
            if call.__name__ == "sample":
                return self.interpret_sample(call, fit=False)
            elif call.__name__ == "observe":
                return self.interpret_observe(call)
            elif call.__name__ == "observe_all":
                return self.interpret_method(call) #see Distribution.observe_all
            elif call.__name__ == "fit": #fit is an alternative interface to sample
                return self.interpret_sample(call, fit=True)
            else:
                # other than sample and observe, we interpret Distribution methods as deterministic
                return self.interpret_method_deterministically(call)
        else:
            return self.interpret_method(call)

    def interpret_method(self, call: "NonCPSCallable") -> 'Continuation':
        # For instance and class methods, we transform and compile the
        # function body like normal but need to ensure the right first
        # argument is passed in.
        # This handles both class and instance methods since
        # call.__self__ refers to the the instance if a method,
        # or the class if a classmethod
        continuation = self.interpret_cps(call.__func__)
        def method_continuation(*args, _cont: 'Continuation'=None, _stack: 'Stack'=None, **kws):
            return lambda : continuation(call.__self__, *args, _cont=_cont, _stack=_stack, **kws)
        return method_continuation

    @method_cache
    def interpret_cps(
        self,
        call : Union['NonCPSCallable', 'CPSCallable']
    ) -> 'Continuation':
        if CPSTransform.is_transformed(call):
            continuation = self.interpret_transformed(call)
        else:
            continuation = self.interpret_generic(call)
        return continuation

    def interpret_transformed(self, call : CPSFunction) -> 'Continuation':
        if (
            (not self._emit_call_entryexit) or \
            (not getattr(call, "_emit_call_entryexit", True)) # hook for debugging
        ):
            return self.interpret_transformed_only(call)
        else:
            return self.interpret_transformed_and_emit_entryexit(call)

    def interpret_transformed_only(self, call : CPSFunction) -> 'Continuation':
        def generic_continuation(*args, _cont: 'Continuation'=None, _stack: 'Stack'=None, **kws):
            return call(*args, **kws, _cps=self, _stack=_stack, _cont=_cont)
        return generic_continuation

    def interpret_transformed_and_emit_entryexit(self, call : CPSFunction) -> 'Continuation':
        def continuation(*args, _cont: 'Continuation'=None, _stack: 'Stack'=None, **kws):
            kws = hashabledict(kws)
            params = dict(f=call, args=args, kwargs=kws, cps=self, stack=_stack)
            def process_controller_instructions(
                run_func: bool = True,
                cached_res: Any = None
            ):
                if run_func:
                    def process_func_result(func_res):
                        return ExitCallState(
                            **params,
                            value=func_res,
                            continuation=lambda : _cont(func_res),
                        )
                    return call(*args, _stack=_stack, _cont=process_func_result, _cps=self, **kws)
                else:
                    return ExitCallState(
                        **params,
                        value=cached_res,
                        continuation=lambda : _cont(cached_res),
                    )
            return EnterCallState(
                **params,
                continuation=process_controller_instructions,
            )
        return continuation

    def interpret_sample(self, call: 'SampleCallable[Element]', fit: bool) -> 'Continuation':
        def sample_continuation(
            _cont: 'Continuation'=None,
            _stack: 'Stack'=None,
            name: 'VariableName'=None,
            initial_value=None,
        ):
            return SampleState(
                continuation=_cont,
                distribution=call.__self__,
                name=name,
                stack=_stack,
                cps=self,
                initial_value=initial_value,
                fit=fit
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
        trans_func = self.non_cps_callable_to_cps_callable(call)
        return self.interpret_transformed(trans_func)

    def non_cps_callable_to_cps_callable(self, call: 'NonCPSCallable') -> CPSFunction:
        assert not CPSTransform.is_transformed(call), "Callable already transformed"
        call_name = self.generate_unique_method_name(call)
        code = self.transform_from_func(call, call_name)
        return self.compile_cps_transformed_code_to_function(code, call, call_name)

    def compile_cps_transformed_code_to_function(
        self,
        code: ast.AST,
        call: 'NonCPSCallable',
        call_name: str
    ) -> CPSFunction:
        compiled_code = self.compile(
            f'{call.__name__}_{hex(id(call)).removeprefix("0x")}.py',
            code,
        )
        context = {
            **call.__globals__,
            **self.get_closure(call),
            CPSTransform.cps_interpreter_name: self,
            "global_store": self.global_store_proxy,
            CPSFunction.__name__: CPSFunction,
            hashabledict.__name__: hashabledict,
            hashablelist.__name__: hashablelist,
        }
        try:
            assert CPSInterpreter._compile_mode is False
            CPSInterpreter._compile_mode = True
            exec(compiled_code, context)
        except SyntaxError as err :
            raise err
        finally:
            CPSInterpreter._compile_mode = False
        trans_func = context[call.__name__] if call_name is None else context[call_name]
        if isinstance(trans_func, (classmethod, staticmethod)):
            # not the most elegant fix but it works
            trans_func = trans_func.__func__
        return trans_func

    def generate_unique_method_name(self, call: 'NonCPSCallable') -> Union[str, None]:
        # this function generates a name that won't override a non-method
        # check if the qualified name indicates it's a class method
        # see https://peps.python.org/pep-3155/
        qualname_suffix = call.__qualname__.split("<locals>.")[-1]
        defined_in_class = len(qualname_suffix.split('.')) > 1
        if defined_in_class:
            return "__"+qualname_suffix.replace('.', '_')
        return None

    def transform_from_func(self, call: 'NonCPSCallable', call_name: str = None) -> ast.AST:
        source = inspect.getsource(call)
        source = textwrap.dedent(source)
        trans_node = ast.parse(source)
        if call_name is not None:
            self.rename_class_method_in_source(trans_node, call_name)
        self.subset_validator(trans_node, source)
        return self.transform(trans_node)

    def rename_class_method_in_source(self, node: ast.Module, name: str):
        assert len(node.body) == 1 and isinstance(node.body[0], ast.FunctionDef), \
            "We assume there's only a single function definition in the source"
        node.body[0].name = name

    def transform(self, trans_node: ast.AST) -> ast.AST:
        self.closure_scope_analysis(trans_node)
        trans_node = self.desugaring_transform(trans_node)
        trans_node = self.setlines_transform(trans_node)
        trans_node = self.cps_transform(trans_node)
        trans_node = self.hashable_collection_transform(trans_node)
        # print(ast.unparse(trans_node))
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

    @staticmethod
    def get_closure(func: 'NonCPSCallable') -> dict:
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


class DescriptorMixIn:
    """
    @private
    A mixin class that provides a descriptor interface for a callable object.

    In Python, a descriptor is an object attribute with "binding behavior",
    meaning that its attribute access is overridden by methods in the descriptor protocol.
    For more details, see [here](https://docs.python.org/3/howto/descriptor.html).

    For our purposes, this mixin class implements a descriptor interface for
    callables that wrap class methods, static methods, and normal methods that is compatible with
    the CPSInterpreter.
    """
    def __init__(self, wrapped_func):
        if inspect.ismethod(wrapped_func):
            raise ValueError("Cannot wrap a method outside the class namespace")
        self.is_classmethod = isinstance(wrapped_func, classmethod)
        self.is_staticmethod = isinstance(wrapped_func, staticmethod)

    def __call__(self, *args, _cont=None, _cps=None, _stack=None, **kws):
        raise NotImplementedError

    def __get__(self, obj, objtype=None):
        # if we wrapped a class method, we return a partial function with cls
        if self.is_classmethod:
            objtype = objtype if objtype is not None else type(obj)
            partial_call = functools.partial(self.__call__, objtype)
            setattr(partial_call, CPSTransform.is_transformed_property, True)
            return partial_call

        # if we wrapped a static method, we return the function itself
        if self.is_staticmethod:
            return self

        # if we wrapped a normal method, we return the function itself or
        # a partial function with obj
        if obj is None: # this means the function is not bound to an instance
            return self
        partial_call = functools.partial(self.__call__, obj)
        setattr(partial_call, CPSTransform.is_transformed_property, True)
        return partial_call

class KeepDeterministicCallable(DescriptorMixIn):
    '''
    @private
    '''
    def __init__(self, func):
        DescriptorMixIn.__init__(self, func)
        self.wrapped_func = func
        if isinstance(func, (classmethod, staticmethod)):
            self.wrapped_func = func.__func__
        functools.update_wrapper(self, func)
        # This ensures that no subsequent compilation happens.
        setattr(self, CPSTransform.is_transformed_property, True)

    def __call__(self, *args, _cont=None, _cps=None, _stack=None, **kws):
        rv = self.wrapped_func(*args, **kws)
        if _cont is None:
            return rv
        else:
            return lambda : _cont(rv)


def keep_deterministic(fn: Callable[..., R]) -> Callable[..., R]:
    '''
    Decorator to interpret a function as deterministic Python.
    Any random sampling in the function will not be targeted for inference.
    Any conditioning will not be incorporated into the inference process.

    This is helpful if the transform slows a function down, if a
    deterministic library is being called, or if a distribution is being
    directly computed.
    '''
    return KeepDeterministicCallable(fn)

def cps_transform_safe_decorator(dec: Callable) -> Callable:
    """
    A higher-order function that wraps a decorator so that it works with functions
    that are CPS transformed or will be CPS transformed.

    Functions are typically evaluated twice in the course of being CPS transformed.
    First, when they are initially declared in Python.
    Second, when being evaluated after being transformed.
    So, a function's decorators would typically be executed twice. However, this decorator
    can be used to mark a decorator so that it will not be executed after being CPS transformed.
    The resulting function will then be the transformed version of decorated function, without
    any further decoration after transformation.
    """
    @keep_deterministic
    def wrapped_decorator(fn=None, *args, **kws):
        if fn is None:
            return functools.partial(wrapped_decorator, *args, **kws)

        # This code mainly supports two cases:
        # 1. We need to CPS transform and wrap a function (e.g., when decorating in the module scope)
        # 2. We only need to wrap a function (e.g., when decorating in a nested function scope)

        # Case 1 is confusing because the transformation and wrapping occur in two different
        # calls to the current code. Specifically, we visit the current
        # code first from the module scope and then transform/wrap it using a CPSInterpreter
        # object. The CPSInterpreter object transforms the source code and then executes the current
        # code a second time on the transformed function (in an "exec" statement). This second
        # call to the current code does not need to transform the function, and this is indicated
        # by a _compile_mode flag associated with the CPSInterpreter class.

        # Case 2 is simpler because we only need to wrap the function. This occurs when the function
        # is nested inside another function that has already been CPS transformed.
        if CPSInterpreter._compile_mode:
            return fn
        wrapped_fn = dec(fn, *args, **kws)
        return wrapped_fn
    wrapped_decorator = functools.wraps(dec)(wrapped_decorator)
    return wrapped_decorator
