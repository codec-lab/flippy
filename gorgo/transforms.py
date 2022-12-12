import ast
import copy

class DesugaringTransform(ast.NodeTransformer):
    """
    This "desugars" the AST by
    - separating out function calls to individual assignments
    - converting <exp> if <cond> else <exp> statements to explicit
    if-else blocks
    """
    # TODO: desugaring can be made more concise
    def __call__(self, rootnode):
        self.new_statements = []
        self.visit(rootnode)
        self.insert_new_statements()
        ast.fix_missing_locations(rootnode)
        rootnode = ast.parse(ast.unparse(rootnode)) #HACK: this might be slow
        return rootnode

    def link_children(self, parent):
        for field, child in ast.iter_fields(parent):
            if isinstance(child, ast.AST):
                child._parent = parent
                child._parent_field = field
                child._parent_field_is_list = False
                child._parent_field_idx = None
            if isinstance(child, list):
                for child_i, list_child in enumerate(child):
                    if isinstance(list_child, ast.AST):
                        list_child._parent = parent
                        list_child._parent_field = field
                        list_child._parent_field_is_list = True
                        list_child._parent_field_idx = child_i

    def generate_name(self, node):
        name = []
        while True:
            if not hasattr(node, "_parent"):
                name.append("")
                break
            if node._parent_field_is_list:
                level_name = f"{node._parent_field}_{node._parent_field_idx}"
            else:
                level_name = f"{node._parent_field}"
            name.append(level_name)
            node = node._parent
        return "__".join(name[::-1])

    def insert_new_statements(self):
        while self.new_statements:
            block, idx, stmt = self.new_statements.pop()
            block.insert(idx, stmt)

    def get_block_blockindex(self, node):
        while True:
            if isinstance(node, ast.stmt):
                assert node._parent_field_is_list
                return getattr(node._parent, node._parent_field), node._parent_field_idx
            node = node._parent

    def generic_visit(self, node):
        self.link_children(node)
        ast.NodeTransformer.generic_visit(self, node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node._parent, ast.Assign):
            return node
        block, idx = self.get_block_blockindex(node)
        return_name = self.generate_name(node)
        preline_assn = ast.Assign(
            targets=[ast.Name(id=return_name, ctx=ast.Store())],
            value=node
        )
        preline_assn._is_preline_assn = True
        self.new_statements.append((
            block,
            idx,
            preline_assn
        ))
        return ast.Name(id=return_name, ctx=ast.Load())

    def visit_IfExp(self, node):
        """
        Convert <exp> if <cond> else <exp> to explicit
        if then else blocks

        Note: this might be less efficient if you have nested IfExp nodes
        since it will evaluate all branches
        """
        self.generic_visit(node)
        block, idx = self.get_block_blockindex(node)
        return_name = self.generate_name(node)
        preline_if = ast.If(
            test=node.test,
            body=[
                ast.Assign(
                    targets=[ast.Name(id=return_name, ctx=ast.Store())],
                    value=node.body
                )
            ],
            orelse=[
                ast.Assign(
                    targets=[ast.Name(id=return_name, ctx=ast.Store())],
                    value=node.orelse
                )
            ],
        )
        self.new_statements.append((
            block,
            idx,
            preline_if
        ))
        return ast.Name(id=return_name, ctx=ast.Load())

class CallWrap_and_Arg_Transform(ast.NodeTransformer):
    """
    Wrap every call with a cps interpreter
    """
    context_name = "<root>"
    def __call__(self, rootnode, call_wrap_name="_cps.interpret"):
        self.call_wrap_name = call_wrap_name
        self.visit(rootnode)
        return rootnode
    def visit_Call(self, node):
        new_address = f"('{self.context_name}', {node.lineno}, )"
        new_node = ast.parse(f'{self.call_wrap_name}(_func)(_address=_address + {new_address})').body[0].value
        new_node.func.args = [node.func]
        new_node.args = node.args
        new_node.keywords = [
            *node.keywords,
            *new_node.keywords
        ]
        return new_node

    def visit_FunctionDef(self, node):
        parent_context_name = self.context_name
        self.context_name = node.name
        cur_keywords = [kw.arg for kw in node.args.kwonlyargs + node.args.args]
        if "_address" not in cur_keywords:
            # TODO: fix addressing scheme for functions
            node.args.kwonlyargs.append(ast.arg(arg='_address'))
            node.args.kw_defaults.append(ast.parse("()").body[0].value)
        if "_cps" not in cur_keywords:
            node.args.kwonlyargs.append(ast.arg(arg='_cps'))
            node.args.kw_defaults.append(ast.parse("_cps").body[0].value)
        self.generic_visit(node)
        self.context_name = parent_context_name
        return node

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

class CPSTransform(ast.NodeTransformer):
    """
    Convert python to a form of continuation passing style.
    """
    final_continuation_name = "_cont"
    is_transformed_property = "_cps_transformed"

    def __call__(self, node):
        return self.visit(node)

    @classmethod
    def is_transformed(cls, func):
        return getattr(func, CPSTransform.is_transformed_property, False)

    def transform_block(self, block):
        self.block = [s for s in block]
        self.new_block = []
        while True:
            self.cur_statement = self.block.pop(0)
            self.visit(self.cur_statement)
            if len(self.block) == 0:
                break
            self.new_block.append(self.cur_statement)
        return self.new_block

    def visit_Call(self, node):
        # construct thunk with trace as arg
        cont_name = f'_cont_{node.lineno}'
        node.keywords.append(ast.keyword(
            arg=self.final_continuation_name,
            value=ast.parse(cont_name).body[0].value
        ))
        ret_node = ast.parse(f'return lambda : _call_()').body[0]
        ret_node.value.body = node

        # create the continuation, recursively
        res_name = f'_res_{node.lineno}'
        cont_node = ast.parse(f"def {cont_name}({res_name}): pass").body[0]
        cont_node.body = [self.cur_statement]
        cont_block = CPSTransform().transform_block(self.block)
        cont_node.body.extend(cont_block)
        self.block = []

        # add the continuation function def, then the return thunk
        self.new_block.append(cont_node)
        self.new_block.append(ret_node)
        return ast.Name(id=res_name, ctx=ast.Load())

    def visit_Return(self, node):
        new_node = ast.parse(f"return lambda : {self.final_continuation_name}(_res)").body[0]
        new_node.value.body.args[0] = node.value
        self.new_block.append(new_node)
        self.block = []

    def visit_FunctionDef(self, node):
        node = copy.deepcopy(node)
        cur_keywords = [kw.arg for kw in node.args.kwonlyargs + node.args.args]
        if self.final_continuation_name not in cur_keywords:
            node.args.kwonlyargs.append(ast.arg(arg=self.final_continuation_name))
            node.args.kw_defaults.append(ast.parse("lambda val: val").body[0].value)
        node.body = CPSTransform().transform_block(node.body)
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

        self.cur_statement = node
        return node

    def visit_If(self, node):
        """
        if <cond>:
            <stmts>
        else:
            <stmts>
        """
        if_branch = node.body + copy.deepcopy(self.block)
        else_branch = node.orelse + self.block
        node.body = CPSTransform().transform_block(if_branch)
        node.orelse = CPSTransform().transform_block(else_branch)
        self.block = []
        self.new_block.append(node)
