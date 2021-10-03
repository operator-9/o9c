import ast
import unittest

from .. import lltransform


STRUCT_SRC = '''import typing

class Node(typing.NamedTuple):
    value: int
    next: 'Node'

def sum(node: Node) -> int:
    result = 0
    while node:
        result += node.value
        node = node.next
    return result
'''


class TestStruct(unittest.TestCase):
    def test_struct_lowering(self):
        transformer = lltransform.LLTransform()
        struct_ast = ast.parse(STRUCT_SRC)
        transformer.visit(struct_ast)
        module_src = str(transformer.module)
        self.assertGreater(len(module_src), 0)


if __name__ == '__main__':
    unittest.main()
