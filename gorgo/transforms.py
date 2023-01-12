import ast
import textwrap
import copy
import contextlib

class NodeTransformer(ast.NodeTransformer):
    def generic_visit(self, node):
        for field, _ in ast.iter_fields(node):
            self.generic_visit_field(node, field)
        return node

    def generic_visit_field(self, node, field):
        # This is the inner loop over fields implemented by `ast.NodeTransformer.generic_visit`.
        # It's useful when a specific order over fields needs to be guaranteed, so we refactor
        # it out here.
        old_value = getattr(node, field)
        if isinstance(old_value, list):
            # This case and next one are both copied from original.
            new_values = []
            for value in old_value:
                if isinstance(value, ast.AST):
                    value = self.visit(value)
                    if value is None:
                        continue
                    elif not isinstance(value, ast.AST):
                        new_values.extend(value)
                        continue
                new_values.append(value)
            old_value[:] = new_values
        elif isinstance(old_value, ast.AST):
            new_node = self.visit(old_value)
            if new_node is None:
                delattr(node, field)
            else:
                setattr(node, field, new_node)

class PythonSubsetValidator(ast.NodeVisitor):
    def __call__(self, node, source):
        filename = 'tmp.py'
        self.errors = []
        self.visit(node)
        if self.errors:
            # HACK For now, we just report the first error.
            node = self.errors[0]
            # Subtract 1 b/c node.lineno is 1-indexed
            code = source.splitlines()[node.lineno - 1]
            # Add 1 b/c node.col_offset is 0-indexed
            offset = node.col_offset + 1
            raise SyntaxError('Found unsupported Python feature.', (filename, node.lineno, offset, code))

    def error(self, node):
        self.errors.append(node)

    visit_Global = error
    visit_Nonlocal = error
    visit_ClassDef = error
    visit_AsyncFunctionDef = error
    visit_AsyncFor = error
    visit_AsyncWith = error
    visit_Await = error
    visit_Yield = error
    visit_YieldFrom = error

class DesugaringTransform(ast.NodeTransformer):
    """
    This "desugars" the AST by
    - separating out function calls to individual assignments
    - converting <exp> if <cond> else <exp> statements to explicit
    if-else blocks
    - converting lambda expressions to function def
    - converting logical and/or statements to equivalent if/else blocks
    - make None function return explicit
    """
    def __call__(self, rootnode):
        self.new_stmt_stack = []
        self.n_temporary_vars = 0
        self.visit(rootnode)
        rootnode = ast.parse(ast.unparse(rootnode)) #HACK: this might be slow
        return rootnode

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            node = self.visit_stmt(node)
        else:
            node = ast.NodeTransformer.generic_visit(self, node)
        return node

    def visit_stmt(self, node):
        self.new_stmt_stack.append([])
        node = ast.NodeTransformer.generic_visit(self, node)
        self.add_statement(node)
        return self.new_stmt_stack.pop()

    def generate_name(self):
        self.n_temporary_vars += 1
        return f"__v{self.n_temporary_vars - 1}"

    def visit_Call(self, node):
        node = ast.NodeTransformer.generic_visit(self, node)
        return_name = self.generate_name()
        assn_node = ast.Assign(
            targets=[ast.Name(id=return_name, ctx=ast.Store())],
            value=node,
            lineno=0
        )
        self.add_statement(assn_node)
        return ast.Name(id=return_name, ctx=ast.Load())

    def visit_IfExp(self, node):
        """
        Convert <exp> if <cond> else <exp> to explicit if then else blocks
        """
        return self.desugar_to_IfElse_block(
            test_expr=node.test,
            if_expr=node.body,
            else_expr=node.orelse,
            test_name = self.generate_name(),
            return_name = self.generate_name()
        )

    def visit_Lambda(self, node):
        def_name = self.generate_name()
        def_node = ast.parse(textwrap.dedent(f"""
        def {def_name}():
            return None
        """)).body[0]
        def_node.args = node.args
        def_node.body[0].value = node.body
        def_node = ast.NodeTransformer.generic_visit(self, def_node)
        self.add_statement(def_node)
        return ast.Name(id=def_name, ctx=ast.Load())
    
    def visit_Return(self, node):
        if node.value is None:
            node.value = ast.Constant(value=None)
        node = self.generic_visit(node)
        return node

    def visit_BoolOp(self, node):
        if len(node.values) > 2:
            leftmost = ast.BoolOp(
                op=type(node.op)(),
                values=node.values[:-1]
            )
            node.values = [leftmost, node.values[-1]]
            return self.visit_BoolOp(node)
        test_name = self.generate_name()
        return_name = self.generate_name()
        if isinstance(node.op, ast.And):
            return self.desugar_to_IfElse_block(
                test_expr=node.values[0],
                if_expr=node.values[1],
                else_expr=ast.Name(id=test_name),
                test_name=test_name,
                return_name=return_name
            )
        elif isinstance(node.op, ast.Or):
            return self.desugar_to_IfElse_block(
                test_expr=node.values[0],
                if_expr=ast.Name(id=test_name),
                else_expr=node.values[1],
                test_name=test_name,
                return_name=return_name
            )
        raise ValueError("BoolOp is neither And nor Or")
    
    def visit_FunctionDef(self, node):
        node_list = self.generic_visit(node)
        # make return value of None function explicit
        # Note: this is required for CPSTransform to work
        if not isinstance(node_list[-1].body[-1], ast.Return):
            return_none = ast.parse("return None").body[0]
            node_list[-1].body.append(return_none)
        return node_list

    def visit_ListComp(self, node):
        '''
        Convert list comprehensions into cps_reduce calls. Later generators correspond to
        nested cps_reduce calls. For the following example input:

        ```
        [
            (x, y)
            for x in range(4)
            if x % 2 == 0
            for y in range(5)
            if x + y < 3
        ]
        ```

        We transform code into the following:

        ```
        cps_reduce(
            lambda __acc, x: (
                __acc + cps_reduce(
                    lambda __acc, y: (
                        __acc + [(x, y)] if all([x + y < 3]) else __acc
                    ),
                    range(5),
                    [],
                ) if all([x % 2 == 0]) else __acc
            ),
            range(4),
            [],
        )
        ```

        '''

        nested = ast.List(elts=[node.elt], ctx=ast.Load())

        for g in node.generators[::-1]:
            assert isinstance(g.target, ast.Name) and isinstance(g.target.ctx, ast.Store), 'Only simple targets are supported.'
            target = g.target.id

            new_node = ast.parse(textwrap.dedent(f'''
            cps_reduce(lambda __acc, {target}: __acc + None if all([]) else __acc, None, [])
            ''')).body[0].value

            new_node.args[0].body.body.right = nested
            new_node.args[0].body.test.args[0].elts = g.ifs
            new_node.args[1] = g.iter

            nested = new_node

        return self.visit(new_node)
    
    def desugar_to_IfElse_block(
        self,
        test_expr,
        if_expr,
        else_expr,
        test_name,
        return_name
    ):
        test_node, if_node = ast.parse(textwrap.dedent(f"""
        {test_name} = test
        if {test_name}:
            {return_name} = if_body
        else:
            {return_name} = else_body
        """)).body
        test_node.value = test_expr
        test_node = self.generic_visit(test_node)
        if not isinstance(test_node, list):
            test_node = [test_node]
        if_node.body[0].value = if_expr
        if_node.orelse[0].value = else_expr
        self.generic_visit(if_node)
        for stmt in [*test_node, if_node]:
            self.add_statement(stmt)
        return ast.Name(id=return_name, ctx=ast.Load())

    def add_statement(self, node):
        self.new_stmt_stack[-1].append(node)

class SetLineNumbers(ast.NodeTransformer):
    """
    Reset line numbers by statement.
    """
    def __call__(self, node, starting_line=0):
        self.cur_line = starting_line
        self.visit(node)
        ast.fix_missing_locations(node)
        return node
    def set_line(self, node):
        node.lineno = self.cur_line
        self.cur_line += 1
    def generic_visit(self, node):
        if isinstance(node, (ast.Expr, ast.stmt)):
            self.set_line(node)
        elif hasattr(node, 'lineno'):
            del node.lineno
        ast.NodeTransformer.generic_visit(self, node)
        return node

class CPSTransform(NodeTransformer):
    """
    Convert python to a form of continuation passing style.
    """
    is_transformed_property = "_cps_transformed"
    func_src_name = "__func_src"
    cps_interpreter_name = "_cps"
    final_continuation_name = "_cont"
    stack_name = "_stack"

    def __init__(self):
        self.scopes = []
        self.remaining_stmts = None
        self.new_block = None
        self.cur_stmt = None

    def __call__(self, node):
        with self.in_scope():
            return self.visit(node)

    @classmethod
    def is_transformed(cls, func):
        return getattr(func, CPSTransform.is_transformed_property, False)

    def visit_Lambda(self, node):
        # Most lambdas are desugared, so this only has to handle the ones we generate,
        # like the default value for _cont of lambda val: val.
        with self.in_scope():
            return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node = self.add_function_src_to_FunctionDef_body(node)
        node = self.add_keyword_to_FunctionDef(node, self.stack_name, "()")
        node = self.add_keyword_to_FunctionDef(node, self.cps_interpreter_name, self.cps_interpreter_name)
        node = self.add_keyword_to_FunctionDef(node, self.final_continuation_name, "lambda val: val")
        with self.in_scope():
            self.visit(node.args)
            node.body = self.transform_block(node.body)
        decorator = ast.parse(f"lambda fn: (fn, setattr(fn, '{self.is_transformed_property}', True))[0]").body[0].value

        # This decorator position makes it the last decorator called.
        # This ensures that the outermost function has `is_transformed_property` set.
        node.decorator_list.insert(0, decorator)
        # This decorator position makes it the first decorator called.
        # This is important so that intermediate decorators can change their behavior
        # the second time that this function is executed.
        # Functions are executed a second time when being evaluted after being CPS-transformed
        # because decorators aren't removed by the transform.
        node.decorator_list.append(decorator)
        return node
    
    def add_keyword_to_FunctionDef(self, node, key, value):
        assert isinstance(node, ast.FunctionDef)
        cur_keywords = [kw.arg for kw in node.args.kwonlyargs + node.args.args]
        assert key not in cur_keywords
        node.args.kwonlyargs.append(ast.arg(arg=key))
        node.args.kw_defaults.append(ast.parse(value).body[0].value)
        return node
    
    def add_function_src_to_FunctionDef_body(self, node):
        assert isinstance(node, ast.FunctionDef)
        ctx_id = repr(ast.unparse(node))
        ctx_id_assn = ast.parse(f"{self.func_src_name} = {ctx_id}").body[0]
        node.body.insert(0, ctx_id_assn)
        return node

    @contextlib.contextmanager
    def in_scope(self):
        self.scopes.append(set())
        try:
            yield
        finally:
            self.scopes.pop()

    def add_name_to_scope(self, name):
        if name in [
            CPSTransform.func_src_name,
            CPSTransform.cps_interpreter_name,
            CPSTransform.final_continuation_name,
            CPSTransform.stack_name,
        ]:
            return
        self.scopes[-1].add(name)

    def visit_arg(self, node):
        self.add_name_to_scope(node.arg)
        return node

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            self.add_name_to_scope(node.id)
        return node

    def visit_Assign(self, node):
        # We make sure to visit values first.
        self.generic_visit_field(node, 'value')
        # Then we visit targets.
        self.generic_visit_field(node, 'targets')
        return node

    def visit_Call(self, node):
        continuation_name = f"_cont_{node.lineno}"
        result_name = f"_res_{node.lineno}"
        scope_name = f"_scope_{node.lineno}"
        names = sorted(self.scopes[-1])
        unpack = '\n'.join(f'''
                if "{name}" in {scope_name}:
                    {name} = {scope_name}["{name}"]
        ''' for name in names) or 'pass'
        code = ast.parse(textwrap.dedent(f'''
            # We capture locals before we enter any other scope (like a lambda or list comprehension)
            _locals = locals()
            {scope_name} = {{name: _locals[name] for name in {names} if name in _locals}}
            def {continuation_name}({result_name}):
                {unpack}
            return lambda : {self.cps_interpreter_name}.interpret(
                _func,
                cont={continuation_name},
                stack={self.stack_name}, 
                func_src={self.func_src_name},
                locals_=_locals,
                lineno={node.lineno}
            )()
        ''')).body
        continuation_node, thunk_node = code[-2:]
        thunk_node.value.body.func.args = [node.func]
        thunk_node.value.body.args = node.args
        thunk_node.value.body.keywords = node.keywords
        continuation_node.body += \
            [self.cur_stmt] + \
            self.transform_block(self.remaining_stmts)
        
        self.remaining_stmts = []
        self.new_block.extend(code)
        return ast.Name(id=result_name, ctx=ast.Load())
        
    def transform_block(self, block):
        # Store current state, so we can recursively transform.
        previous = self.remaining_stmts, self.new_block, self.cur_stmt

        self.remaining_stmts = block
        self.new_block = []
        while True:
            self.cur_stmt = self.remaining_stmts.pop(0)
            # cur_stmt gets added or updated explicitly in visit methods
            self.visit(self.cur_stmt)  
            if len(self.remaining_stmts) == 0:
                break
            self.new_block.append(self.cur_stmt)
            
        block[:] = self.new_block

        # Restore previous state.
        self.remaining_stmts, self.new_block, self.cur_stmt = previous

        return block

    def visit_If(self, node):
        """
        if <cond>:
            <stmts>
        else:
            <stmts>
        """
        if_branch = node.body + copy.deepcopy(self.remaining_stmts)
        else_branch = node.orelse + self.remaining_stmts
        node.body = self.transform_block(if_branch)
        node.orelse = self.transform_block(else_branch)
        self.remaining_stmts = []
        
        # add node explicitly
        self.new_block.append(node)

    def visit_Return(self, node):
        new_node = ast.parse(f"return lambda : {self.final_continuation_name}(_res)").body[0]
        new_node.value.body.args[0] = node.value
        
        # add node explicitly
        self.new_block.append(new_node)
