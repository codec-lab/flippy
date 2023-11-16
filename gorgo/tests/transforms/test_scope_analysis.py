import ast
import textwrap
from gorgo.transforms import *
import pytest

def _analyze(src):
    src = textwrap.dedent(src)
    node = ast.parse(src)
    ClosureScopeAnalysis()(node, src)

def test_scope_analysis():
    # Defined before
    _analyze('''
    def fn():
        x = 3
        def inner():
            return x
        return inner
    ''')

    # Defined after
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            def inner():
                return x
            x = 3
            return inner
        ''')
    assert 'defined before' in str(err)

    # Defined before, nested
    _analyze('''
    def fn():
        x = 3
        def outer():
            def inner():
                return x
            return inner
        outer()
    ''')

    # Defined after, nested
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            def outer():
                def inner():
                    return x
                return inner
            x = 3
            outer()
        ''')
    assert 'defined before' in str(err)

    # Mutating local
    _analyze('''
    def fn():
        def inner():
            x = 3
            x = 4
            return x
        return inner
    ''')

    # Defined before, mutated before
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            x = 3
            x = x + 4
            def inner():
                return x
            return inner
        ''')
    assert 'immutable' in str(err)

    # Defined before, mutated after
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            x = 3
            def inner():
                return x
            x = 4
            return inner
        ''')
    assert 'immutable' in str(err)

    # Defined after, mutated after
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            def inner():
                return x
            x = 3
            x = 4
            return inner
        ''')
    assert 'immutable' in str(err)

def test_scope_analysis_AnnAssign():
    # Defined before
    _analyze('''
    def fn():
        x: int = 3
        def inner():
            return x
        return inner
    ''')

    # Defined after
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            def inner():
                return x
            x: int = 3
            return inner
        ''')
    assert 'defined before' in str(err)

def test_scope_analysis_FunctionDef():
    # Mutating non-local
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            def abc(): pass
            def abc(): pass
            def inner():
                return abc()
            return inner
        ''')
    assert 'immutable' in str(err)

def test_scope_analysis_AugAssign():
    # Mutating local
    _analyze('''
    def fn():
        def inner():
            x = 3
            x += 4
            return x
        return inner
    ''')

    # Mutating non-local
    with pytest.raises(SyntaxError) as err:
        _analyze('''
        def fn():
            x = 3
            x += 4
            def inner():
                return x
            return inner
        ''')
    assert 'immutable' in str(err)

def test_scope_analysis_Comp():
    for begin, end, elt in [
        # List comprehension
        ('[', ']', lambda x: x),
        # Set comprehension
        ('{', '}', lambda x: x),
        # Dict comprehension
        ('{', '}', lambda x: f'{x}:{x}'),
    ]:
        # Test that comprehension has a separate scope.
        _analyze(f'''
        def fn():
            x = 3
            y = {begin}{elt('x')} for x in range(3){end}
            def inner():
                return x
            return inner
        ''')

        # Tests when overwriting variables in comprehension
        # NOTE: These are ok because they never have a nested scope.
        _analyze(f'''
        def fn():
            x = [3, 4]
            return {begin}{elt('x')} for x in x{end}
        ''')
        _analyze(f'''
        def fn():
            x = [3, 4]
            return {begin}{elt('x')} for x in x for x in x{end}
        ''')

        # Ensure reference to nonlocal mutated elt raises
        with pytest.raises(SyntaxError) as err:
            _analyze(f'''
            def fn():
                c = [3, 4]
                c = [5]
                def inner():
                    return {begin}{elt('c')} for x in [3, 4]{end}
                return inner
            ''')
        assert 'immutable' in str(err)

        # Ensure reference to nonlocal mutated if raises
        with pytest.raises(SyntaxError) as err:
            _analyze(f'''
            def fn():
                c = [3, 4]
                c = [5]
                def inner():
                    return {begin}{elt('x')} for x in [3, 4] if c{end}
                return inner
            ''')
        assert 'immutable' in str(err)

        # Reference to mutated iter in local scope raises
        # NOTE: This seems conservative, but seems tricky to handle elegantly with
        # multiple comprehensions.
        with pytest.raises(SyntaxError) as err:
            _analyze(f'''
            def fn():
                c = [3, 4]
                c = [5]
                x = {begin}{elt('x')} for x in c{end}
            ''')
        assert 'immutable' in str(err)

        # Ensure reference to nonlocal mutated iter raises
        with pytest.raises(SyntaxError) as err:
            _analyze(f'''
            def fn():
                c = [3, 4]
                c = [5]
                def inner():
                    return {begin}{elt('x')} for x in c{end}
                return inner
            ''')
        assert 'immutable' in str(err)

        # Ensure reference to nonlocal mutated iter raises, for second comprehension
        with pytest.raises(SyntaxError) as err:
            _analyze(f'''
            def fn():
                c = [3, 4]
                c = [5]
                def inner():
                    return {begin}{elt('x')} for a in [1] for x in c{end}
                return inner
            ''')
        assert 'immutable' in str(err)

    # Ensure reference to nonlocal mutated elt raises
    # Special case for dict key VS value
    for elt in ['x: c', 'c: x']:
        with pytest.raises(SyntaxError) as err:
            _analyze(f'''
            def fn():
                c = 0
                c = 1
                def inner(): return {{{elt} for x in []}}
            ''')
        assert 'immutable' in str(err)
