import ast
import functools
import inspect
import textwrap
import types
from typing import Tuple, Callable
from gorgo.core import ReturnState, SampleState, ObserveState, InitialState
from gorgo.core import StochasticPrimitive, ObservationStatement, StackFrame
from gorgo.transforms import DesugaringTransform, \
    SetLineNumbers, CPSTransform
from gorgo.funcutils import method_cache
import linecache
import types

class CPSInterpreter:
    lambda_func_name = "__lambda_func__"
    def __init__(self):
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
        context = {**func.__globals__, **self.get_closure(func), "_cps": self}
        try:
            exec(code, context)
        except SyntaxError as err :
            raise err
        if self.is_lambda_function(func):
            trans_func = context[self.lambda_func_name]
        else:
            trans_func = context[func.__name__]
        def wrapper_generic(*args, _cont=lambda v: v, _stack=None, **kws):
            return trans_func(*args, **kws, _cps=self, _stack=_stack, _cont=_cont)
        return wrapper_generic

    def transform_from_func(self, func):
        if self.is_lambda_function(func):
            source = self.get_lambda_source(func)
        else:
            source = textwrap.dedent(inspect.getsource(func))
        trans_node = ast.parse(source)
        return self.transform(trans_node)
    
    @classmethod
    def get_lambda_source(cls, func):
        # Based on http://xion.io/post/code/python-get-lambda-code.html
        # We get all lambda functions defined in the source and then
        # compare the bytecode length of each if there are multiple
        
        full_src = textwrap.dedent(inspect.getsource(func))
        src = full_src[full_src.find("lambda"):]
        while True:
            if len(src) <= 0:
                raise ParsingError(f"Unable to parse:\n{full_src}")
            try:
                trans_node = ast.parse(src)
                break
            except SyntaxError:
                src = src[:-1]
        lambda_nodes = FindLambdaNodes()(trans_node)
        
        if len(lambda_nodes) == 1:
            lambda_src = ast.unparse(lambda_nodes[0])
            return f"{cls.lambda_func_name} = {lambda_src}"
        
        # HACK: we can try to see which lambda expression matches
        # the function by comparing bytecode and other function
        # meta-data, but it might not be unique
        def get_lambda_hash(fn):
            return (
                len(fn.__code__.co_code),
                fn.__code__.co_varnames,
                fn.__code__.co_argcount,
                fn.__code__.co_posonlyargcount,
                fn.__code__.co_kwonlyargcount,
            )
        true_lambda_hash = get_lambda_hash(func)
        lambda_hash_to_src = {}
        for lambda_node in lambda_nodes:
            lambda_src = ast.unparse(lambda_node)
            new_lambda = eval(lambda_src)
            lambda_hash = get_lambda_hash(new_lambda)
            if lambda_hash == true_lambda_hash and lambda_hash in lambda_hash_to_src:
                raise ParsingError(f"Unable to uniquely parse lambda from source:\n{full_src}")
            lambda_hash_to_src[lambda_hash] = lambda_src
        if true_lambda_hash in lambda_hash_to_src:
            lambda_src = lambda_hash_to_src[true_lambda_hash]
            return f"{cls.lambda_func_name} = {lambda_src}"
        raise ParsingError(f"Unable to parse lambda from source:\n{full_src}")
    
    @classmethod
    def is_lambda_function(cls, func):
        return isinstance(func, types.LambdaType) and func.__name__ == "<lambda>"

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

class FindLambdaNodes(ast.NodeVisitor):
    def __call__(self, node):
        self.lambda_nodes = []
        self.visit(node)
        return self.lambda_nodes
    
    def visit_Lambda(self, node):
        self.lambda_nodes.append(node)
        self.generic_visit(node)

class ParsingError(Exception): pass