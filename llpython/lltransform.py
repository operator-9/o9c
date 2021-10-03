import ast
import importlib
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
    # TODO: ast.MatMult
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
    # TODO: ast.Pow
    ast.LShift : {
        ir.IntType: ir.IRBuilder.shl,
    },
    ast.RShift : {
        ir.IntType: ir.IRBuilder.ashr,
    },
    ast.BitOr : {
        ir.IntType: ir.IRBuilder.or_,
    },
    ast.BitXor : {
        ir.IntType: ir.IRBuilder.xor,
    },
    ast.BitAnd : {
        ir.IntType: ir.IRBuilder.and_,
    },
    ast.FloorDiv : {
        ir.IntType: ir.IRBuilder.udiv,
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

    # ______________________________________________________________________
    # Utilities

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

    # ______________________________________________________________________
    # Statements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.function is not None:
            raise NotImplementedError('closures unsupportd')
        fname = node.name
        args = [self.handle_arg(arg) for arg in node.args.args]
        ftype = ir.FunctionType(
            self.handle_annotation(node.returns),
            [arg.lltype for arg in args]
        )
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

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        raise NotImplementedError('async functions')

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        raise NotImplementedError('classes')

    def visit_Return(self, node: ast.Return) -> Any:
        return self.builder.ret(self.visit(node.value))

    def visit_Delete(self, node: ast.Delete) -> Any:
        raise NotImplementedError('delete')

    def visit_Assign(self, node: ast.Assign) -> Any:
        if self.function is None:
            raise NotImplementedError('assignment of global')
        llvalue = self.visit(node.value)
        target_count = len(node.targets)
        if target_count > 1:
            raise NotImplementedError('tuple unpacking in assignment')
        else:
            assert target_count == 1
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                raise NotImplementedError('complex assignment')
            llvalue.name = target.id
            self.namespace[target.id] = llvalue
        return llvalue

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        raise NotImplementedError('augmented assignment')

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            pymod = importlib.import_module(alias.name)
            pymod_name = alias.asname if alias.asname is not None else alias.name
            # FIXME: Mixing ir.Value's and Python objects in this namespace.
            self.namespace[pymod_name] = pymod

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        raise NotImplementedError('import from')

    # ______________________________________________________________________
    # Expressions

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            return self.lookup(node.id)
        raise NotImplementedError(f'unhandled name usage "{ast.dump(node)}"')

    def visit_BinOp(self, node: ast.BinOp) -> ir.Value:
        left = self.visit(node.left)
        left_type = type(left.type)
        right = self.visit(node.right)
        right_type = type(right.type)
        op_type = type(node.op)
        if left_type != right_type:
            raise TypeError(f'incompatible operands "{op_type} {left_type}, {right_type}"')
        return BINOPS[op_type][left_type](self.builder, left, right)

    def visit_Constant(self, node: ast.Constant) -> ir.Value:
        return {
            bool: TYPES['bool'],
            int: TYPES['int'],
            float: TYPES['float'],
        }[type(node.value)](node.value)


if __name__ == '__main__':
    transformer = LLTransform()
    src = '''def fpadd(n0: float, n1: float) -> float:
    return n0 + n1

def thingy(n0: int, n1: int) -> int:
    x = n0 << 2
    y = n1 << 2
    return x + y
    '''
    tree = ast.parse(src)
    print(ast.dump(tree, indent=2))
    transformer.visit(tree)
    print(transformer.module)
