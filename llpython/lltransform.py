import ast
from typing import Any, Dict, List, NamedTuple, Optional

from llvmlite import ir


class Arg(NamedTuple):
    name: str
    lltype: ir.Type


TYPES = { # FIXME: Assumes a 64-bit world...
    'bool': ir.IntType(1),
    'int8' : ir.IntType(8),
    'int16' : ir.IntType(16),
    'int32' : ir.IntType(32),
    'int64' : ir.IntType(64),
    'cfloat' : ir.FloatType(),
    'double' : ir.DoubleType(),
    'int': ir.IntType(64),
    'float': ir.DoubleType(),
}

BINOPS = {
    ast.Add : {
        ir.IntType: ir.IRBuilder.add,
        ir.FloatType: ir.IRBuilder.fadd,
        ir.DoubleType: ir.IRBuilder.fadd,
    },
    ast.Sub : {
        ir.IntType: ir.IRBuilder.sub,
        ir.FloatType: ir.IRBuilder.fsub,
        ir.DoubleType: ir.IRBuilder.fsub,
    },
    ast.Mult : {
        ir.IntType: ir.IRBuilder.mul,
        ir.FloatType: ir.IRBuilder.fmul,
        ir.DoubleType: ir.IRBuilder.fmul,
    },
    ast.Div : {
        ir.IntType: ir.IRBuilder.udiv,
        ir.FloatType: ir.IRBuilder.fdiv,
        ir.DoubleType: ir.IRBuilder.fdiv,
    },
    ast.Mod : {
        ir.IntType: ir.IRBuilder.urem,
        ir.FloatType: ir.IRBuilder.frem,
        ir.DoubleType: ir.IRBuilder.frem,
    },
}


class LLTransform(ast.NodeVisitor):
    function: Optional[ir.Function]
    namespace: Dict[str, ir.Value]
    namespaces: List[Dict[str, ir.Value]]
    builder: Optional[ir.IRBuilder]

    def __init__(self, module: ir.Module = None, *args, **kws):
        super().__init__(*args, **kws)
        self.module = module if module is not None else ir.Module()
        self.function = None
        self.namespace = {}
        self.namespaces = []
        self.builder = None

    def push_namespace(self):
        self.namespaces.append(self.namespace)
        self.namespace = {}

    def pop_namespace(self):
        self.namespace = self.namespaces.pop()

    def lookup(self, name: str) -> ir.Value:
        if name in self.namespace:
            return self.namespace[name]
        for ns in self.namespaces[::-1]:
            if name in ns:
                return ns[name]
        raise NameError(name)

    def handle_annotation(self, annotation: ast.expr) -> ir.Type:
        if isinstance(annotation, ast.Name) and annotation.id in TYPES:
            return TYPES[annotation.id]
        raise TypeError(f'unknown type annotation, "{ast.dump(annotation)}"')

    def handle_arg(self, arg: ast.arg) -> Arg:
        return Arg(arg.arg, self.handle_annotation(arg.annotation))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.function is not None:
            raise NotImplementedError('closures unsupportd')
        fname = node.name
        args = [self.handle_arg(arg) for arg in node.args.args]
        ftype = ir.FunctionType(self.handle_annotation(node.returns), [arg.lltype for arg in args])
        func = ir.Function(self.module, ftype, fname)
        self.function = func
        self.namespace[fname] = self.function
        self.push_namespace()
        for index, arg in enumerate(args):
            llarg = self.function.args[index]
            llarg.name = arg.name
            self.namespace[arg.name] = llarg
        entry = func.append_basic_block()
        self.builder = ir.IRBuilder(entry)
        for statement in node.body:
            self.visit(statement)
        self.pop_namespace()
        self.function = None
        return func

    def visit_Name(self, node: ast.Name) -> ir.Value:
        if isinstance(node.ctx, ast.Load):
            return self.lookup(node.id)
        raise NotImplementedError(f'unhandled name usage "{ast.dump(node)}"')

    def visit_BinOp(self, node: ast.BinOp) -> ir.Value:
        left = self.visit(node.left)
        left_type = type(left.type)
        right = self.visit(node.right)
        right_type = type(right.type)
        assert left_type == right_type
        op_type = type(node.op)
        return BINOPS[op_type][left_type](self.builder, left, right)

    def visit_Return(self, node: ast.Return) -> Any:
        return self.builder.ret(self.visit(node.value))


if __name__ == '__main__':
    transformer = LLTransform()
    src = '''def fpadd(n0: float, n1: float) -> float:
    return n0 + n1
    '''
    tree = ast.parse(src)
    print(ast.dump(tree, indent=2))
    transformer.visit(tree)
    print(transformer.module)
