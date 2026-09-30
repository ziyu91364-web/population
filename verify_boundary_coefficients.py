#!/usr/bin/env python3
"""Exact coefficient verification over Q[i,z,z^{-1}], using only stdlib.

The first calculation uses z=exp(i*phi) in the square Hamiltonian.
The second uses z=1/c in the inverse dephasing of the real three-path.
No floating-point arithmetic or symbolic-algebra package is used.
"""
from fractions import Fraction


class Laurent:
    # Keys are (integer power of z, power of i reduced to 0 or 1).
    def __init__(self, value=0):
        if isinstance(value, Laurent):
            self.p = value.p.copy()
        elif isinstance(value, dict):
            self.p = {k: Fraction(v) for k, v in value.items() if v}
        else:
            self.p = {(0, 0): Fraction(value)} if value else {}

    def __add__(self, other):
        other = Laurent(other)
        out = self.p.copy()
        for k, v in other.p.items():
            out[k] = out.get(k, Fraction(0)) + v
        return Laurent(out)

    __radd__ = __add__

    def __neg__(self):
        return Laurent({k: -v for k, v in self.p.items()})

    def __sub__(self, other):
        return self + (-Laurent(other))

    def __rsub__(self, other):
        return Laurent(other) + (-self)

    def __mul__(self, other):
        other = Laurent(other)
        out = {}
        for (power1, imag1), v1 in self.p.items():
            for (power2, imag2), v2 in other.p.items():
                imag = imag1 + imag2
                key = (power1 + power2, imag % 2)
                val = v1 * v2 * (-1 if imag >= 2 else 1)
                out[key] = out.get(key, Fraction(0)) + val
        return Laurent(out)

    __rmul__ = __mul__

    def __eq__(self, other):
        return self.p == Laurent(other).p

    def __bool__(self):
        return bool(self.p)

    def __repr__(self):
        if not self.p:
            return '0'
        return ' + '.join(f'{v}*z^{power}' + ('*i' if imag else '')
                          for (power, imag), v in sorted(self.p.items()))


I = Laurent({(0, 1): 1})
Z = Laurent({(1, 0): 1})
ZI = Laurent({(-1, 0): 1})


def zero(rows, cols):
    return [[Laurent() for _ in range(cols)] for _ in range(rows)]


def add(a, b):
    return [[u + v for u, v in zip(ar, br)] for ar, br in zip(a, b)]


def neg(a):
    return [[-u for u in row] for row in a]


def mul(a, b):
    result = zero(len(a), len(b[0]))
    for i, row in enumerate(a):
        for k, u in enumerate(row):
            if u:
                for j, v in enumerate(b[k]):
                    if v:
                        result[i][j] = result[i][j] + u*v
    return result


def left_diagonal(diagonal, a):
    return [[d*u for u in row] for d, row in zip(diagonal, a)]


def blocks(h):
    n = len(h)
    pop = [i*(n+1) for i in range(n)]
    coh = [i for i in range(n*n) if i not in pop]
    a = zero(n*n, n*n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    comm = (h[i][k] if ell == j else Laurent())
                    comm -= h[ell][j] if i == k else Laurent()
                    a[i+n*j][k+n*ell] = -I*comm
    B = [[a[i][j] for j in coh] for i in pop]
    C = [[a[i][j] for j in pop] for i in coh]
    D = [[a[i][j] for j in coh] for i in coh]
    return B, C, D, coh


def coefficients(h, inverse_rates=None, maximum=4):
    B, C, D, coh = blocks(h)
    r = inverse_rates or [Laurent(1)]*len(coh)
    V = [left_diagonal(r, C)]
    for n in range(1, maximum+1):
        nonlinear = zero(len(C), len(C[0]))
        for i in range(n-1):
            nonlinear = add(nonlinear, mul(mul(V[i], B), V[n-2-i]))
        V.append(left_diagonal(r, add(mul(D, V[n-1]), neg(nonlinear))))
    return [mul(B, v) for v in V]


def main():
    h = [[Laurent(v) for v in row] for row in
         [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]]
    h[0][3], h[3][0] = ZI, Z
    K = coefficients(h)
    cosine_twice = Z+ZI
    for i in range(4):
        for j in range(4):
            expected = (72*cosine_twice-80 if i == j else
                        120*cosine_twice-48 if (i-j) % 4 == 2 else
                        64-96*cosine_twice)
            assert K[4][i][j] == expected
            assert K[1][i][j] == 0
            assert K[3][i][j] == 0
    assert K[2][0][2] == 6*cosine_twice-4
    print('Square K2_13:', K[2][0][2])
    print('Square K4_13:', K[4][0][2])
    print('At cos(phi)=1/3: K4_13 =', 120*Fraction(2, 3)-48)

    h = [[Laurent(v) for v in row] for row in [[0,1,0],[1,0,1],[0,1,0]]]
    _, _, _, coh = blocks(h)
    r = [Z if {j % 3, j // 3} == {0, 2} else Laurent(1) for j in coh]
    K = coefficients(h, inverse_rates=r)
    assert K[2][0][2] == 2*Z-4
    assert K[4][0][2] == 8*Z*Z+36*Z-64
    assert K[4][2][0] == K[4][0][2]
    print('Weighted path K2_13:', K[2][0][2])
    print('Weighted path K4_13:', K[4][0][2])
    print('At c=1/2: K4_13 =', 8*Fraction(2)**2+36*Fraction(2)-64)
    print('All exact coefficient assertions passed.')


if __name__ == '__main__':
    main()
