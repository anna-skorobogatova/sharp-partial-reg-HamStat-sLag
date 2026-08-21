#!/usr/bin/env python3
"""Independent SymPy audit of the Cartan-cubic reduction.

This script does not import the interval validator.  It starts from the
polynomial P, differentiates it symbolically, forms the full 5-by-5 Hessian of
u=r^2 f(P/r^3) at a rational parametrization of the orbit section, and checks
the block Hessian, determinant reduction, printed D/N polynomials, endpoint
identity, and corrected real-part formula.
"""
from __future__ import annotations

import platform
import sympy as sp


def check(label: str, expr) -> None:
    value = sp.cancel(sp.together(expr))
    if value != 0:
        value = sp.factor(value)
    if value != 0:
        raise AssertionError(f"FAIL: {label}: {value}")
    print(f"PASS: {label}")


def main() -> None:
    print(f"Python {platform.python_version()}")
    print(f"SymPy {sp.__version__}")

    x1, x2, z1, z2, z3 = sp.symbols("x1 x2 z1 z2 z3", real=True)
    X = sp.Matrix([x1, x2, z1, z2, z3])
    sqrt3 = sp.sqrt(3)
    P = (
        x1**3
        + sp.Rational(3, 2) * x1 * (z1**2 + z2**2 - 2*z3**2 - 2*x2**2)
        + sp.Rational(3, 2) * sqrt3 * (x2*z1**2 - x2*z2**2 + 2*z1*z2*z3)
    )
    r2 = sum(v**2 for v in X)
    gradP = sp.Matrix([sp.diff(P, v) for v in X])
    hessP = sp.hessian(P, X)
    check("Cartan eikonal identity |grad P|^2=9r^4", gradP.dot(gradP) - 9*r2**2)
    check("Cartan harmonicity Delta P=0", sum(sp.diff(P, v, 2) for v in X))

    Q = sp.Matrix([
        [2*x1, sqrt3*z1, sqrt3*z2],
        [sqrt3*z1, -x1+sqrt3*x2, sqrt3*z3],
        [sqrt3*z2, sqrt3*z3, -x1-sqrt3*x2],
    ])
    check("P=(1/2)det Q", P - sp.Rational(1, 2)*Q.det())
    check("r^2=(1/6)tr(Q^2)", r2 - sp.Rational(1, 6)*sp.trace(Q*Q))

    # Rational parametrization c=cos(theta), d=sin(theta).
    t = sp.symbols("t", real=True)
    c = (1-t**2)/(1+t**2)
    d = 2*t/(1+t**2)
    s = sp.cancel(c**3 - 3*c*d**2)
    sigma = sp.cancel(3*d - 4*d**3)
    check("unit-circle parametrization", c**2+d**2-1)
    check("s^2+sigma^2=1", s**2+sigma**2-1)

    subs_section = {x1:c, x2:d, z1:0, z2:0, z3:0}
    xsec = sp.Matrix([c,d,0,0,0])
    gp = gradP.subs(subs_section)
    hp = hessP.subs(subs_section)
    grad_s = gp - 3*s*xsec
    hess_s = hp - 3*(xsec*gp.T + gp*xsec.T) - 3*s*sp.eye(5) + 15*s*(xsec*xsec.T)

    f, y, q = sp.symbols("f y q", real=True)
    Hu = 2*f*sp.eye(5) + 2*y*(xsec*grad_s.T + grad_s*xsec.T) + q*(grad_s*grad_s.T) + y*hess_s
    er = sp.Matrix([c,d,0,0,0])
    et = sp.Matrix([-d,c,0,0,0])
    E = sp.Matrix.hstack(er,et,sp.eye(5)[:,2],sp.eye(5)[:,3],sp.eye(5)[:,4])
    Hb = sp.simplify(E.T*Hu*E)

    p = -3*sigma*y
    mu1 = 3*(c+sqrt3*d-s)
    mu2 = 3*(c-sqrt3*d-s)
    mu3 = -3*(2*c+s)
    expected = sp.diag(1,1,1,1,1)*0
    expected[0,0] = 2*f
    expected[0,1] = p
    expected[1,0] = p
    expected[1,1] = 2*f + 9*(1-s**2)*q - 9*s*y
    expected[2,2] = 2*f+y*mu1
    expected[3,3] = 2*f+y*mu2
    expected[4,4] = 2*f+y*mu3
    for i in range(5):
        for j in range(5):
            check(f"Hessian block entry ({i+1},{j+1})", Hb[i,j]-expected[i,j])

    check("sum mu_j=-9s", mu1+mu2+mu3+9*s)
    check("sum mu_i mu_j=-27(1-s^2)", mu1*mu2+mu1*mu3+mu2*mu3+27*(1-s**2))
    check("product mu_j=27s(1-s^2)", mu1*mu2*mu3-27*s*(1-s**2))

    I = sp.I
    K = 1+2*I*f
    T_product = sp.expand((K+I*y*mu1)*(K+I*y*mu2)*(K+I*y*mu3))
    T_formula = K**3 + 3*p**2*K - 3*I*s*y*(3*K**2+p**2)
    check("transverse product T", T_product-T_formula)

    det_direct = sp.factor((sp.eye(5)+I*Hb).det())
    det_formula = T_formula*(K**2+p**2+9*I*K*((1-s**2)*q-s*y))
    check("full 5x5 determinant reduction", det_direct-det_formula)

    # Printed real polynomials, checked with s independent and p^2=9(1-s^2)y^2.
    S = sp.symbols("s", real=True)
    K0 = 1+2*I*f
    p2 = 9*(1-S**2)*y**2
    T0 = K0**3 + 3*p2*K0 - 3*I*S*y*(3*K0**2+p2)
    D_complex = sp.expand(sp.re(K0*T0))
    N_complex = sp.expand(sp.im((K0**2+p2)*T0))
    D_print = (
        16*f**4-72*f**3*S*y+108*f**2*S**2*y**2-108*f**2*y**2-24*f**2
        -54*f*S**3*y**3+54*f*S*y**3+54*f*S*y-27*S**2*y**2+27*y**2+1
    )
    N_print = (
        32*f**5-144*f**4*S*y+288*f**3*S**2*y**2-288*f**3*y**2-80*f**3
        -432*f**2*S**3*y**3+432*f**2*S*y**3+216*f**2*S*y
        +486*f*S**4*y**4-972*f*S**2*y**4-216*f*S**2*y**2+486*f*y**4+216*f*y**2+10*f
        -243*S**5*y**5+486*S**3*y**5+108*S**3*y**3-243*S*y**5-108*S*y**3-9*S*y
    )
    check("printed D polynomial", D_complex-D_print)
    check("printed N polynomial", N_complex-N_print)

    H_endpoint = sp.expand(N_print.subs(S,1)-9*y*D_print.subs(S,1))
    check("endpoint determinant compatibility", H_endpoint-sp.im(K0**3*(K0-9*I*y)**2))
    check("endpoint derivative identity d_y H=-18D", sp.diff(H_endpoint,y)+18*D_print.subs(S,1))

    # Corrected real-part identity after imposing the ODE.
    Tbar = sp.conjugate(T0)
    W = (K0**2+p2)*T0
    Z = K0*T0
    D0 = sp.re(Z)
    N0 = sp.im(W)
    det_on_ode = W-I*(N0/D0)*Z
    rhs = (T0*Tbar)*((K0*sp.conjugate(K0))+p2)/D0
    check("corrected Re(det)=|T|^2(|K|^2+p^2)/D", sp.re(det_on_ode)-rhs)

    print("SYMBOLIC AUDIT VERIFIED")


if __name__ == "__main__":
    main()
