
import pytest
from flippy.hashable import hashabledict, hashablelist, hashableset

def test_hashableset():
    hs = hashableset([1,2,3])
    s = set([1,2,3])
    assert hs == s, "Equality depends only on contents"
    hs2 = hashableset(s)
    assert hash(hs) == hash(hs2), "Hash equality depends only on contents"
    assert type(hs | {4}) == hashableset, "Adding a set to a hashableset returns a hashableset"
    assert type({4} | hs ) == hashableset, "Adding a set to a hashableset returns a hashableset"
    assert type(hs2 & {4}) == hashableset, "Intersecting a hashableset with a set returns a hashableset"
    assert type({4} & hs2) == hashableset, "Intersecting a hashableset with a set returns a hashableset"
    assert {hs: 1} == {hs2: 1}, "Hashableset is hashable"
    assert {hs: 1} != {hashableset(hs2 | {4}): 2}, "Hashableset is hashable"

    with pytest.raises(TypeError):
        hs.update([4])
    with pytest.raises(TypeError):
        hs.intersection_update([4])
    with pytest.raises(TypeError):
        hs |= [4]
    with pytest.raises(TypeError):
        hs &= [4]
    with pytest.raises(TypeError):
        hs ^= [4]
    with pytest.raises(TypeError):
        hs.difference_update([4])
    with pytest.raises(TypeError):
        hs.symmetric_difference_update([4])
    with pytest.raises(TypeError):
        hs.add(4)
    with pytest.raises(TypeError):
        hs.remove(4)
    with pytest.raises(TypeError):
        hs.discard(4)
    with pytest.raises(TypeError):
        hs.pop()
    with pytest.raises(TypeError):
        hs.clear()

def test_hashablelist():
    hl = hashablelist([1,2,3])
    l = [1,2,3]
    assert hl == l, "Equality depends only on contents"
    hl2 = hashablelist(l)
    assert hash(hl) == hash(hl2), "Hash equality depends only on contents"
    assert type(hl + [4]) == hashablelist, "Adding a list to a hashablelist returns a hashablelist"
    assert type([4] + hl) == hashablelist, "Adding a list to a hashablelist returns a hashablelist"
    assert type(hl2 * 3) == hashablelist, "Multiplying a hashablelist returns a list"
    assert type(3*hl2) == hashablelist, "Multiplying a hashablelist returns a list"
    assert {hl: 1} == {hl2: 1}, "Hashablelist is hashable"
    assert {hl: 1} != {hashablelist(hl2 + [4]): 2}, "Hashablelist is hashable"

    # Test immutability guardrails
    with pytest.raises(TypeError):
        hl[0] = 3
    with pytest.raises(TypeError):
        del hl[0]
    with pytest.raises(TypeError):
        hl.append(4)
    with pytest.raises(TypeError):
        hl.insert(0, 4)
    with pytest.raises(TypeError):
        hl.pop()
    with pytest.raises(TypeError):
        hl.remove()
    with pytest.raises(TypeError):
        hl.clear()
    with pytest.raises(TypeError):
        hl += [4]
    with pytest.raises(TypeError):
        hl.sort()
    with pytest.raises(TypeError):
        hl *= 2
    with pytest.raises(TypeError):
        hl.reverse()

def test_hashabledict():
    hd = hashabledict({'a': 1, 'b': 2})
    d = {'b': 2, 'a': 1}
    assert hd == d, "Equality depends only on keys and values, not order or hashability"
    hd2 = hashabledict(d)
    assert hash(hd) == hash(hd2), "Hashes are equal for equal keys and values"
    assert type(hd | {'c': 3}) == hashabledict, "Union of hashabledict is hashabledict"
    assert type({'c': 3} | hd) == hashabledict, "Union of hashabledict is hashabledict"
    assert {hd: 1} == {hd: 1}, "hashabledict is hashable"
    assert {hd: 1} != {hashabledict(hd | {'c': 3}): 1}

    # Test immutability guardrails
    with pytest.raises(TypeError):
        hd['c'] = 3
    with pytest.raises(TypeError):
        del hd['a']
    with pytest.raises(TypeError):
        hd.update({'c': 3})
    with pytest.raises(TypeError):
        hd.pop('a')
    with pytest.raises(TypeError):
        hd.popitem()
    with pytest.raises(TypeError):
        hd.clear()
    with pytest.raises(TypeError):
        hd.setdefault('c', 3)
    with pytest.raises(TypeError):
        hd.fromkeys('abc') # this uses .__setitem__
    with pytest.raises(TypeError):
        hd |= {'c': 3}

def test_recursive_coercing():
    hash(hashabledict({'a': (), 'k': {}}))
    hash(hashabledict({'a': [1,2,3], 'b': {'c': {4,5,6}}, 'd': {'e': 7}}))
    hash(hashablelist([1,2,3, [4,5,6], {7,8,9}, {'a': 10, 'b': {}}]))
