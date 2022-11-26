import ast
import inspect
from gorgo.core import ProgramState, ReturnMessage, \
    StartingMessage, SampleMessage, ObserveMessage, \
    StochasticPrimitive, ObservationStatement
from gorgo.transforms import DesugaringTransform, \
    CallWrap_and_Arg_Transform, SetLineNumbers, CPSTransform

class CPSInterpreter:
    def __init__(self):
        self.desugaring_transform = DesugaringTransform()
        self.call_transform = CallWrap_and_Arg_Transform()
        self.setlines_transform = SetLineNumbers()
        self.cps_transform = CPSTransform()
    
    def initial_program_state(self, function):
        interpreted_function = self.interpret(function)
        def return_continuation(value):
            return ProgramState(
                continuation=None,
                message=ReturnMessage(
                    address=None,
                    value=value
                ),
                is_returned=True
            )
        def program_continuation(*args, **kws):
            return interpreted_function(
                *args,
                _cont=return_continuation,
                **kws
            )
        return ProgramState(
            continuation=program_continuation,
            message=StartingMessage(
                address=()
            ),
        )
        
    def interpret(self, call):
        if isinstance(call, type):
            return self.interpret_class(call)
        if hasattr(call, "__self__"):
            if isinstance(call.__self__, StochasticPrimitive) and call.__name__ == "sample":
                return self.interpret_sample(call)
        if isinstance(call, ObservationStatement):
            return self.interpret_observation(call)
        return self.interpret_generic(call)
    
    def interpret_sample(self, call):
        def sample_wrapper(_address, _cont):
            return ProgramState(
                continuation=_cont,
                message=SampleMessage(
                    address=_address,
                    distribution=call
                ),
                is_returned=False
            )
        return sample_wrapper
    
    def interpret_observation(self, func):
        def observation_wrapper(*args, _address=None, _cont=None, **kws):
            return ProgramState(
                continuation=lambda : _cont(None),
                message=ObserveMessage(
                    address=_address,
                    distribution=args[0] if len(args) >= 1 else kws['distribution'],
                    value=args[1] if len(args) >= 2 else kws['value']
                )
            )
        return observation_wrapper
    
    def interpret_class(self, cls):
        def class_wrapper(*args, _address=None, _cont=None, **kws):
            return lambda :  _cont(cls(*args, **kws))
        return class_wrapper
    
    def interpret_generic(self, func):
        trans_node = ast.parse(inspect.getsource(func))
        trans_node = self.desugaring_transform(trans_node)
        trans_node = self.call_transform(trans_node)
        trans_node = self.setlines_transform(trans_node)
        trans_node = self.cps_transform(trans_node)
        trans_source = ast.unparse(trans_node)
        # print(trans_source)
        local_context = {**self.get_closure(func), "_cps": self}
        try:
            exec(trans_source, func.__globals__, local_context)
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