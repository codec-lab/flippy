from flippy.transforms import PythonSubsetValidator
import ast
import textwrap
import pytest

def _validate(source):
    v = PythonSubsetValidator()
    source = textwrap.dedent(source)
    node = ast.parse(source)
    v(node, source)

def _invalid(source, exception_code):
    with pytest.raises(SyntaxError) as err:
        _validate(source)

    assert 'Found unsupported Python feature.' in str(err)

    # Now check exception content to make sure it references code.
    _, exc, tb = err._excinfo
    assert exception_code in exc.text
    # Can make this check more extensive by borrowing from test_interpreter.py

def test_validator():
    _invalid('def fn():\n  global x', 'global x')
    _invalid('def fn():\n  nonlocal x', 'nonlocal x')

    _invalid('class X(object):\n  pass', 'class X')
    _invalid('async def fn():\n  pass', 'async def fn')

    _invalid('async for x in range(3): pass', 'async for x')
    _invalid('async with x: pass', 'async with')
    _invalid('with x: pass', 'with')

    if hasattr(ast, 'Match'):
        _invalid('match None:\n\tcase _: pass', 'match')

    _invalid('''
    try:
        pass
    except:
        pass
    ''', 'try')
    if hasattr(ast, 'TryStar'):
        _invalid('''
        try:
            pass
        except* Exception as e:
            pass
        ''', 'try')

    _invalid('import x', 'import')
    _invalid('from x import x', 'import')

    _invalid('await x', 'await')
    _invalid('yield x', 'yield x')
    _invalid('yield from x', 'yield from x')
    _invalid('if x := 3: pass', ':=')

def test_assignment():
    _validate('x = 3')
    _validate('del x')
    _validate('x += 3')
    _validate('x: int')
    _validate('x: int = 3')

    _validate('[x, y] = range(2)')
    _validate('(x, y) = range(2)')
    _validate('[[x, y], z] = [[0, 1], 2]')

    # Testing setting from attribute to make sure ctx=Load() works.
    _validate('x = sum.__name__')

    for lhs in ['x.f', 'x[0]']:
        _invalid(f'{lhs} = 3', f'{lhs} = 3')
        _invalid(f'del {lhs}', f'del {lhs}')
        _invalid(f'{lhs} += 3', f'{lhs} += 3')
        _invalid(f'{lhs}: int = 3', f'{lhs}: int = 3')
        _invalid(f'[{lhs}] = [3]', f'[{lhs}] = [3]')
