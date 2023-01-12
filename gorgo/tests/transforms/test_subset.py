from gorgo.transforms import PythonSubsetValidator
import ast
import textwrap
import pytest

def _check(source, exception_code):
    v = PythonSubsetValidator()
    source = textwrap.dedent(source)
    node = ast.parse(source)

    with pytest.raises(SyntaxError) as err:
        v(node, source)

    assert 'Found unsupported Python feature.' in str(err)

    # Now check exception content to make sure it references code.
    _, exc, tb = err._excinfo
    assert exception_code in exc.text
    # Can make this check more extensive by borrowing from test_interpreter.py

def test_validator():
    _check('def fn():\n  global x', 'global x')
    _check('def fn():\n  nonlocal x', 'nonlocal x')
    _check('class X(object):\n  pass', 'class X')
    _check('async def fn():\n  pass', 'async def fn')
    _check('async for x in range(3): pass', 'async for x')
    _check('async with x: pass', 'async with')
    _check('await x', 'await x')
    _check('yield x', 'yield x')
    _check('yield from x', 'yield from x')
