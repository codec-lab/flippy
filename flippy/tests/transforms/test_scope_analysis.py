import ast
import textwrap
from flippy.transforms import *
import pytest

def _analyze(src):
    src = textwrap.dedent(src)
    node = ast.parse(src)
    a = ClosureScopeAnalysis()
    a(node, src)
    return a, node

def _analyze_with_expected_error(msg, src):
    with pytest.raises(SyntaxError) as err:
        _analyze(src)
    assert msg in str(err)

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

def test_scope_analysis_shadowing():
    _analyze('''
    def fn():
        x = 3
        x = 3
        def inner():
            x = 4
            x = 4
            return x
        return inner() + x
    ''')

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

def test_scope_analysis_If():
    template = lambda s: f'''
    def fn():
        before = 3
        if True:
            both1 = 1
            body1 = 1
            body2 = 2
            body2 = 3
            body2orelse1 = 4
            body2orelse1 = 5
            body1orelse2 = 7
        else:
            both1 = 2
            orelse1 = 2
            orelse2 = 3
            orelse2 = 4
            body2orelse1 = 7
            body1orelse2 = 8
            body1orelse2 = 9
        after = 8
        def inner():
            {s}
    '''
    a, node = _analyze(template('pass'))
    assert a.complete_scope_map[node.body[0]].ns == collections.Counter(dict(
        both1=1,
        body1=1,
        body2=2,
        orelse1=1,
        orelse2=2,
        body2orelse1=2,
        body1orelse2=2,
        inner=1,
        before=1,
        after=1,
    ))
    # Variables used once, in either or both branches, are ok.
    _analyze(template('return both1'))
    _analyze(template('return body1'))
    _analyze(template('return orelse1'))
    _analyze_with_expected_error('immutable', template('return body2'))
    _analyze_with_expected_error('immutable', template('return orelse2'))
    _analyze_with_expected_error('immutable', template('return body1orelse2'))
    _analyze_with_expected_error('immutable', template('return body2orelse1'))
