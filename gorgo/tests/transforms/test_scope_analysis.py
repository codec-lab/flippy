import ast
import textwrap
from gorgo.transforms import *
import pytest

def test_scope_analysis():
    def _analyze(src):
        node = ast.parse(textwrap.dedent(src))
        ScopeAnalysis()(node)

    # Defined before
    _analyze('''
    def fn():
        x = 3
        def inner():
            return x
        return inner
    ''')

    # Defined after
    with pytest.raises(AssertionError) as err:
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
    with pytest.raises(AssertionError) as err:
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

    # Defined before, mutated after
    with pytest.raises(AssertionError) as err:
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
    with pytest.raises(AssertionError) as err:
        _analyze('''
        def fn():
            def inner():
                return x
            x = 3
            x = 4
            return inner
        ''')
    assert 'immutable' in str(err)
