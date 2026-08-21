#!/usr/bin/env python3
"""Independent exact-rational interval validation of the shooting argument.

This implementation is intentionally separate from validator_core.py.  It uses
only Python's standard-library Fraction type, exact rational interval
arithmetic, independently generated Taylor polynomials, a different singular
start, a different regular step rule, a different Taylor order, and a different
subdivision count.  Transcendental endpoint slopes are enclosed by rational
alternating-series bounds (Machin's formula for pi, then sine/cosine bounds).

The proof architecture is the same mathematical a-posteriori argument:
  * singular Briot--Bouquet contraction on x in [0,1/250];
  * residual/Jacobian/Gronwall continuation to x=1;
  * strict D positivity on every tube;
  * endpoint sign enclosures for A_minus and A_plus.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
import json
import math
import platform
from decimal import Decimal, getcontext, localcontext, ROUND_FLOOR, ROUND_CEILING
import sys
import time
from typing import Iterable, Sequence

A_MINUS = Q("0.66922990609204402834133929821078023454")
A_CENTER = Q("0.66922990609204402834133930821078023454")
A_PLUS = Q("0.66922990609204402834133931821078023454")

# Independent numerical choices (different from the Arb run).
LOCAL_H = Q(1, 250)                 # 0.004, rather than 0.01
LOCAL_ORDER = 24                    # rather than 40
LOCAL_CELLS = 128                   # rather than 64
LOCAL_TRIAL = Q(1, 10**10)
REGULAR_ORDER = 26                  # rather than 30
REGULAR_CELLS = 12                  # rather than 16
REGULAR_TRIAL = Q(1, 10**12)        # rather than 1e-10
MAX_STEP = Q(1, 100)                # 0.01, with the different x/3 growth rule
COEFF_PLACES = 80
STATE_PLACES = 70
BOUND_PLACES = 70
TRANS_TERMS = 70
getcontext().prec = 120


@dataclass(frozen=True)
class IV:
    lo: Q
    hi: Q

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError((self.lo, self.hi))

    @staticmethod
    def point(x: Q | int) -> "IV":
        x = Q(x)
        return IV(x, x)

    def __add__(self, other) -> "IV":
        other = as_iv(other)
        return IV(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self) -> "IV":
        return IV(-self.hi, -self.lo)

    def __sub__(self, other) -> "IV":
        return self + (-as_iv(other))

    def __rsub__(self, other) -> "IV":
        return as_iv(other) - self

    def __mul__(self, other) -> "IV":
        other = as_iv(other)
        vals = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return IV(min(vals), max(vals))

    __rmul__ = __mul__

    def reciprocal(self) -> "IV":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError(f"interval contains zero: {self}")
        return IV(1 / self.hi, 1 / self.lo) if self.lo > 0 else IV(1 / self.hi, 1 / self.lo)

    def __truediv__(self, other) -> "IV":
        return self * as_iv(other).reciprocal()

    def __rtruediv__(self, other) -> "IV":
        return as_iv(other) / self

    def __pow__(self, n: int) -> "IV":
        if n < 0:
            return (self.reciprocal()) ** (-n)
        if n == 0:
            return IV.point(1)
        if n % 2 == 1:
            return IV(self.lo**n, self.hi**n)
        vals = [self.lo**n, self.hi**n]
        return IV(0 if self.lo <= 0 <= self.hi else min(vals), max(vals))

    def abs_upper(self) -> Q:
        return max(abs(self.lo), abs(self.hi))

    def midpoint(self) -> Q:
        return (self.lo + self.hi) / 2

    def radius(self) -> Q:
        return (self.hi - self.lo) / 2

    def subset_of(self, other: "IV") -> bool:
        return other.lo <= self.lo and self.hi <= other.hi

    def hull(self, other: "IV") -> "IV":
        return IV(min(self.lo, other.lo), max(self.hi, other.hi))


def as_iv(x) -> IV:
    return x if isinstance(x, IV) else IV.point(Q(x))


def sym(radius: Q) -> IV:
    if radius < 0:
        raise ValueError(radius)
    return IV(-radius, radius)


def ceil_decimal(x: Q, places: int = BOUND_PLACES) -> Q:
    if x < 0:
        return -floor_decimal(-x, places)
    scale = 10**places
    return Q((x.numerator * scale + x.denominator - 1) // x.denominator, scale)


def floor_decimal(x: Q, places: int = BOUND_PLACES) -> Q:
    if x < 0:
        return -ceil_decimal(-x, places)
    scale = 10**places
    return Q((x.numerator * scale) // x.denominator, scale)


def round_decimal(x: Q, places: int) -> Q:
    scale = 10**places
    sign = -1 if x < 0 else 1
    a = abs(x)
    n = a.numerator * scale
    q, r = divmod(n, a.denominator)
    if 2*r >= a.denominator:
        q += 1
    return Q(sign*q, scale)


def fmt(x: Q, digits: int = 35) -> str:
    # Display only; no proof decision uses this conversion.
    return f"{float(x):.{min(digits,16)}e}" if x and (abs(x) < Q(1,10**8) or abs(x) > 10**8) else str(round(float(x), 16))


def qstr(x: Q) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# Monomial term lists (coefficient, power of s, power of f, power of y).
D_TERMS = [
    (1,0,0,0), (16,0,4,0), (-72,1,3,1), (108,2,2,2),
    (-108,0,2,2), (-24,0,2,0), (-54,3,1,3), (54,1,1,3),
    (54,1,1,1), (-27,2,0,2), (27,0,0,2),
]
N_TERMS = [
    (32,0,5,0), (-144,1,4,1), (288,2,3,2), (-288,0,3,2),
    (-80,0,3,0), (-432,3,2,3), (432,1,2,3), (216,1,2,1),
    (486,4,1,4), (-972,2,1,4), (-216,2,1,2), (486,0,1,4),
    (216,0,1,2), (10,0,1,0), (-243,5,0,5), (486,3,0,5),
    (108,3,0,3), (-243,1,0,5), (-108,1,0,3), (-9,1,0,1),
]


def eval_terms(terms, s, f, y):
    out = 0
    for c, ps, pf, py in terms:
        out = out + c * (s**ps) * (f**pf) * (y**py)
    return out


def eval_terms_derivative(terms, variable: str, s, f, y):
    index = {"s":1, "f":2, "y":3}[variable]
    out = 0
    for term in terms:
        c, ps, pf, py = term
        powers = [ps,pf,py]
        p = powers[index-1]
        if p == 0:
            continue
        powers[index-1] -= 1
        out = out + c*p*(s**powers[0])*(f**powers[1])*(y**powers[2])
    return out


def dn(s, f, y):
    return eval_terms(D_TERMS,s,f,y), eval_terms(N_TERMS,s,f,y)


def h_and_derivatives(x, f, y, regular: bool):
    s = 1-x
    D = eval_terms(D_TERMS,s,f,y)
    N = eval_terms(N_TERMS,s,f,y)
    Df = eval_terms_derivative(D_TERMS,"f",s,f,y)
    Dy = eval_terms_derivative(D_TERMS,"y",s,f,y)
    Nf = eval_terms_derivative(N_TERMS,"f",s,f,y)
    Ny = eval_terms_derivative(N_TERMS,"y",s,f,y)
    H = N - 9*s*y*D
    Hf = Nf - 9*s*y*Df
    Hy = Ny - 9*s*(D+y*Dy)
    den_factor = 9*(x*(2-x) if regular else (2-x))
    den = den_factor*D
    denf = den_factor*Df
    deny = den_factor*Dy
    Ff = (Hf*den-H*denf)/(den**2)
    Fy = (Hy*den-H*deny)/(den**2)
    return H/den, Ff, Fy, D


# Exact polynomial arithmetic, coefficients in ascending order.
def ptrim(a: Sequence[Q]) -> list[Q]:
    a = list(a)
    while len(a)>1 and a[-1]==0:
        a.pop()
    return a


def padd(a,b,order: int|None=None):
    n=max(len(a),len(b))
    if order is not None: n=min(n,order+1)
    out=[Q(0)]*n
    for i in range(n):
        out[i]=(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0)
    return ptrim(out)


def pneg(a): return [-x for x in a]
def psub(a,b,order=None): return padd(a,pneg(b),order)
def pscale(a,c,order=None): return ptrim([(x*c) for x in (a[:order+1] if order is not None else a)])


def pmul(a,b,order: int|None=None):
    n=len(a)+len(b)-1
    if order is not None: n=min(n,order+1)
    out=[Q(0)]*n
    for i,ai in enumerate(a):
        if i>=n: break
        for j,bj in enumerate(b):
            if i+j>=n: break
            out[i+j]+=ai*bj
    return ptrim(out)


def ppow(a,n,order: int|None=None):
    out=[Q(1)]
    base=list(a)
    while n:
        if n&1: out=pmul(out,base,order)
        n//=2
        if n: base=pmul(base,base,order)
    return out


def pinv(a,order):
    if a[0]==0: raise ZeroDivisionError("series constant is zero")
    out=[Q(0)]*(order+1)
    out[0]=1/a[0]
    for k in range(1,order+1):
        out[k]=-sum((a[j] if j<len(a) else 0)*out[k-j] for j in range(1,k+1))/a[0]
    return ptrim(out)


def pdiv(a,b,order): return pmul(a,pinv(b,order),order)

def pder(a): return [Q(i)*a[i] for i in range(1,len(a))] or [Q(0)]


def peval(a,x):
    out=0
    for c in reversed(a): out=out*x+c
    return out


def peval_iv(a,x:IV):
    out=IV.point(0)
    for c in reversed(a): out=out*x+c
    return out


def pl1(a,h:Q):
    out=Q(0); hp=Q(1)
    for c in a:
        out += abs(c)*hp
        hp *= h
    return out


def terms_poly(terms,S,F,Y,order: int|None=None):
    maxs=max(t[1] for t in terms); maxf=max(t[2] for t in terms); maxy=max(t[3] for t in terms)
    Sp=[ppow(S,k,order) for k in range(maxs+1)]
    Fp=[ppow(F,k,order) for k in range(maxf+1)]
    Yp=[ppow(Y,k,order) for k in range(maxy+1)]
    out=[Q(0)]
    for c,ps,pf,py in terms:
        term=pmul(pmul(Sp[ps],Fp[pf],order),Yp[py],order)
        out=padd(out,pscale(term,Q(c),order),order)
    return out


def dn_poly(S,F,Y,order: int|None=None):
    return terms_poly(D_TERMS,S,F,Y,order), terms_poly(N_TERMS,S,F,Y,order)


def polynomial_hash(F,Y):
    payload="F\n"+"\n".join(qstr(x) for x in F)+"\nY\n"+"\n".join(qstr(x) for x in Y)
    return sha256(payload.encode("ascii")).hexdigest()


# Rigorous transcendental bounds using only the standard-library Decimal
# module with directed rounding.  The resulting finite decimals are converted
# exactly to Fraction endpoints before entering the proof computation.
DEC_PREC = 150

@dataclass(frozen=True)
class DI:
    lo: Decimal
    hi: Decimal
    def __post_init__(self):
        if self.lo > self.hi: raise ValueError((self.lo,self.hi))
    @staticmethod
    def point(x:Decimal): return DI(x,x)
    def __add__(self,o):
        o=as_di(o)
        with localcontext() as c:
            c.prec=DEC_PREC; c.rounding=ROUND_FLOOR; lo=self.lo+o.lo
        with localcontext() as c:
            c.prec=DEC_PREC; c.rounding=ROUND_CEILING; hi=self.hi+o.hi
        return DI(lo,hi)
    __radd__=__add__
    def __neg__(self): return DI(-self.hi,-self.lo)
    def __sub__(self,o): return self+(-as_di(o))
    def __rsub__(self,o): return as_di(o)-self
    def __mul__(self,o):
        o=as_di(o)
        vals=[]
        for a in (self.lo,self.hi):
            for b in (o.lo,o.hi):
                with localcontext() as c:
                    c.prec=DEC_PREC; c.rounding=ROUND_FLOOR; vlo=a*b
                with localcontext() as c:
                    c.prec=DEC_PREC; c.rounding=ROUND_CEILING; vhi=a*b
                vals.append((vlo,vhi))
        return DI(min(v[0] for v in vals),max(v[1] for v in vals))
    __rmul__=__mul__
    def reciprocal(self):
        if self.lo<=0<=self.hi: raise ZeroDivisionError
        with localcontext() as c:
            c.prec=DEC_PREC; c.rounding=ROUND_FLOOR; lo=Decimal(1)/self.hi
        with localcontext() as c:
            c.prec=DEC_PREC; c.rounding=ROUND_CEILING; hi=Decimal(1)/self.lo
        return DI(lo,hi)
    def __truediv__(self,o): return self*as_di(o).reciprocal()
    def __rtruediv__(self,o): return as_di(o)/self


def as_di(x):
    if isinstance(x,DI): return x
    if isinstance(x,Decimal): return DI.point(x)
    return DI.point(Decimal(x))


def q_to_di(x:Q)->DI:
    with localcontext() as c:
        c.prec=DEC_PREC; c.rounding=ROUND_FLOOR; lo=Decimal(x.numerator)/Decimal(x.denominator)
    with localcontext() as c:
        c.prec=DEC_PREC; c.rounding=ROUND_CEILING; hi=Decimal(x.numerator)/Decimal(x.denominator)
    return DI(lo,hi)


def atan_small_di(x:DI,n:int=TRANS_TERMS)->DI:
    if x.lo<0 or x.hi>=1: raise ValueError(x)
    total=DI.point(Decimal(0)); term=x; x2=x*x
    for k in range(n):
        add=term/Decimal(2*k+1)
        total=total+add if k%2==0 else total-add
        term=term*x2
    rem=term/Decimal(2*n+1)
    return total + (DI(Decimal(0),rem.hi) if n%2==0 else DI(-rem.hi,Decimal(0)))


def pi_di()->DI:
    return Decimal(16)*atan_small_di(q_to_di(Q(1,5)))-Decimal(4)*atan_small_di(q_to_di(Q(1,239)))

PI_DI=pi_di()
PI=IV(Q(str(PI_DI.lo)),Q(str(PI_DI.hi)))


def atan_positive_di(x:DI)->DI:
    if x.lo<0: raise ValueError(x)
    if x.lo>=1:
        t=(x-Decimal(1))/(x+Decimal(1))
        return PI_DI/Decimal(4)+atan_small_di(t)
    if x.hi<=Decimal('0.5'):
        return atan_small_di(x)
    t=(Decimal(1)-x)/(Decimal(1)+x)
    return PI_DI/Decimal(4)-atan_small_di(t)


def sin_di(x:DI,n:int=TRANS_TERMS)->DI:
    if x.lo<0: raise ValueError(x)
    total=DI.point(Decimal(0)); term=x; x2=x*x
    for k in range(n):
        total=total+term if k%2==0 else total-term
        term=term*x2/Decimal((2*k+2)*(2*k+3))
    return total + (DI(Decimal(0),term.hi) if n%2==0 else DI(-term.hi,Decimal(0)))


def cos_di(x:DI,n:int=TRANS_TERMS)->DI:
    if x.lo<0: raise ValueError(x)
    total=DI.point(Decimal(0)); term=DI.point(Decimal(1)); x2=x*x
    for k in range(n):
        total=total+term if k%2==0 else total-term
        term=term*x2/Decimal((2*k+1)*(2*k+2))
    return total + (DI(Decimal(0),term.hi) if n%2==0 else DI(-term.hi,Decimal(0)))


def tan_di(x:DI)->DI:
    c=cos_di(x)
    if c.lo<=0: raise ArithmeticError('cosine not positive')
    return sin_di(x)/c


def endpoint_B_interval(A:IV)->IV:
    # Monotonicity permits endpoint evaluation.
    alo=atan_positive_di(Decimal(2)*q_to_di(A.lo))
    ahi=atan_positive_di(Decimal(2)*q_to_di(A.hi))
    atanI=DI(alo.lo,ahi.hi)
    z=Decimal(3)/Decimal(2)*atanI
    t=tan_di(z)
    twoA=DI(Decimal(2)*q_to_di(A.lo).lo,Decimal(2)*q_to_di(A.hi).hi)
    B=(twoA+t)/Decimal(9)
    return IV(Q(str(B.lo)),Q(str(B.hi)))

# Independent Taylor construction.  Decimal arithmetic is used only to
# choose polynomial centers; every proof inequality is subsequently checked
# with exact rational arithmetic on the rationalized polynomials.
def dpoly_add(a,b,order):
    out=[Decimal(0)]*(order+1)
    for i in range(order+1):
        out[i]=(a[i] if i<len(a) else Decimal(0))+(b[i] if i<len(b) else Decimal(0))
    return out

def dpoly_scale(a,c,order):
    c=Decimal(c); return [(a[i] if i<len(a) else Decimal(0))*c for i in range(order+1)]

def dpoly_mul(a,b,order):
    out=[Decimal(0)]*(order+1)
    for i in range(min(len(a),order+1)):
        for j in range(min(len(b),order+1-i)):
            out[i+j]+=a[i]*b[j]
    return out

def dpoly_pow(a,n,order):
    out=[Decimal(1)]+[Decimal(0)]*order; base=list(a)+[Decimal(0)]*max(0,order+1-len(a))
    while n:
        if n&1: out=dpoly_mul(out,base,order)
        n//=2
        if n: base=dpoly_mul(base,base,order)
    return out

def dpoly_inv(a,order):
    out=[Decimal(0)]*(order+1); out[0]=Decimal(1)/a[0]
    for k in range(1,order+1):
        out[k]=-sum((a[j] if j<len(a) else Decimal(0))*out[k-j] for j in range(1,k+1))/a[0]
    return out

def dpoly_div(a,b,order): return dpoly_mul(a,dpoly_inv(b,order),order)

def dterms_poly(terms,S,F,Y,order):
    maxs=max(t[1] for t in terms); maxf=max(t[2] for t in terms); maxy=max(t[3] for t in terms)
    Sp=[dpoly_pow(S,k,order) for k in range(maxs+1)]
    Fp=[dpoly_pow(F,k,order) for k in range(maxf+1)]
    Yp=[dpoly_pow(Y,k,order) for k in range(maxy+1)]
    out=[Decimal(0)]*(order+1)
    for c,ps,pf,py in terms:
        term=dpoly_mul(dpoly_mul(Sp[ps],Fp[pf],order),Yp[py],order)
        out=dpoly_add(out,dpoly_scale(term,c,order),order)
    return out

def q_to_dec(x:Q)->Decimal: return Decimal(x.numerator)/Decimal(x.denominator)

def dec_to_q(x:Decimal,places:int)->Q:
    quantum=Decimal(1).scaleb(-places)
    return Q(str(x.quantize(quantum)))

def singular_taylor(Ac:Q,y0:Q,order:int=LOCAL_ORDER):
    f=[Decimal(0)]*(order+2); y=[Decimal(0)]*(order+1)
    f[0]=q_to_dec(Ac); y[0]=q_to_dec(y0)
    _,_,base_gy_q,_=h_and_derivatives(Q(0),Ac,y0,False)
    base_gy=q_to_dec(base_gy_q)
    for k in range(1,order+1):
        f[k]=-y[k-1]/Decimal(k)
        S=[Decimal(1),Decimal(-1)]
        D=dterms_poly(D_TERMS,S,f[:k+1],y[:k+1],k)
        N=dterms_poly(N_TERMS,S,f[:k+1],y[:k+1],k)
        H=dpoly_add(N,dpoly_scale(dpoly_mul(dpoly_mul(S,y[:k+1],k),D,k),-9,k),k)
        den=dpoly_scale(dpoly_mul([Decimal(2),Decimal(-1)],D,k),9,k)
        G=dpoly_div(H,den,k)
        y[k]=(G[k]/(Decimal(k)-base_gy)).quantize(Decimal(1).scaleb(-COEFF_PLACES))
    Y=[dec_to_q(v,COEFF_PLACES) for v in y]
    F=[Ac]+[-Y[j]/Q(j+1) for j in range(order+1)]
    return F,Y,base_gy_q


def regular_taylor(x0:Q,f0:Q,y0:Q,order:int=REGULAR_ORDER):
    f=[Decimal(0)]*(order+2); y=[Decimal(0)]*(order+1)
    f[0]=q_to_dec(f0); y[0]=q_to_dec(y0)
    x0d=q_to_dec(x0)
    for k in range(order):
        f[k+1]=-y[k]/Decimal(k+1)
        X=[x0d,Decimal(1)]; S=[Decimal(1)-x0d,Decimal(-1)]
        D=dterms_poly(D_TERMS,S,f[:k+1],y[:k+1],k)
        N=dterms_poly(N_TERMS,S,f[:k+1],y[:k+1],k)
        H=dpoly_add(N,dpoly_scale(dpoly_mul(dpoly_mul(S,y[:k+1],k),D,k),-9,k),k)
        den=dpoly_scale(dpoly_mul(dpoly_mul(X,[Decimal(2)-x0d,Decimal(-1)],k),D,k),9,k)
        Qs=dpoly_div(H,den,k)
        y[k+1]=(Qs[k]/Decimal(k+1)).quantize(Decimal(1).scaleb(-COEFF_PLACES))
    Y=[dec_to_q(v,COEFF_PLACES) for v in y]
    F=[f0]+[-Y[j]/Q(j+1) for j in range(order+1)]
    return F,Y


def path_bounds(F,Y,x0:Q,h:Q,f_radius:Q,y_radius:Q,cells:int,regular:bool):
    Dmin=None; Mf=Q(0); My=Q(0); L=Q(0)
    for cell in range(cells):
        a=h*cell/cells; b=h*(cell+1)/cells
        t=IV(a,b); x=IV.point(x0)+t
        fb=peval_iv(F,t)+sym(f_radius); yb=peval_iv(Y,t)+sym(y_radius)
        val,df,dy,D=h_and_derivatives(x,fb,yb,regular)
        if D.lo<=0:
            raise ArithmeticError(f"D not positive: x0={x0}, cell={cell}, Dlo={D.lo}")
        Dmin=D.lo if Dmin is None else min(Dmin,D.lo)
        if regular:
            L=max(L,1+df.abs_upper()+dy.abs_upper())
        else:
            Mf=max(Mf,df.abs_upper()); My=max(My,(dy+1).abs_upper())
    return {"Dmin":Dmin,"Mf":Mf,"My":My,"L":L}


def exp_upper(z:Q)->Q:
    if z<0: raise ValueError(z)
    N=70
    term=Q(1); total=Q(1)
    for k in range(1,N+1):
        term=term*z/k; total+=term
    nxt=term*z/(N+1)
    ratio=z/(N+2)
    if ratio>=1: raise ArithmeticError("exp tail ratio")
    return ceil_decimal(total+nxt/(1-ratio),BOUND_PLACES)


def residual_polynomials(F,Y,x0:Q,regular:bool):
    X=[x0,Q(1)] if regular else [Q(0),Q(1)]
    S=[1-x0,Q(-1)]
    D,N=dn_poly(S,F,Y,None)
    H=psub(N,pscale(pmul(pmul(S,Y),D),9))
    if regular:
        den=pscale(pmul(pmul(X,[2-x0,Q(-1)]),D),9)
        residual=psub(pmul(pder(Y),den),H)
    else:
        den=pscale(pmul([Q(2),Q(-1)],D),9)
        residual=psub(pmul(pmul(X,pder(Y)),den),H)
    return residual


def validate_local(Alo:Q,Ahi:Q,label:str):
    Ac=(Alo+Ahi)/2; Ar=(Ahi-Alo)/2; AI=IV(Alo,Ahi)
    B=endpoint_B_interval(AI)
    y0=round_decimal(B.midpoint(),COEFF_PLACES)
    F,Y,basegy=singular_taylor(Ac,y0)
    coarse=path_bounds(F,Y,Q(0),LOCAL_H,Ar+LOCAL_H*LOCAL_TRIAL,LOCAL_TRIAL,LOCAL_CELLS,False)
    q=ceil_decimal(LOCAL_H*coarse["Mf"]+coarse["My"])
    if q>=1: raise ArithmeticError(f"local q={q}")
    residual=residual_polynomials(F,Y,Q(0),False)
    num=pl1(residual,LOCAL_H)
    den=9*(2-LOCAL_H)*coarse["Dmin"]
    Rs=ceil_decimal(num/den)
    V=ceil_decimal((coarse["Mf"]*Ar+Rs)/(1-q))
    if V>=LOCAL_TRIAL: raise ArithmeticError(f"local tail {V}")
    y0box=IV(y0-V,y0+V)
    if not B.subset_of(y0box):
        raise ArithmeticError(f"B interval not contained: B={B}, box={y0box}")
    if coarse["My"]>=1: raise ArithmeticError("G_y not negative")
    ferr=ceil_decimal(Ar+LOCAL_H*V)
    narrow=path_bounds(F,Y,Q(0),LOCAL_H,ferr,V,LOCAL_CELLS,False)
    state=(peval(F,LOCAL_H)+sym(ferr),peval(Y,LOCAL_H)+sym(V))
    record={
        "A":[qstr(Alo),qstr(Ahi)],"B":[qstr(B.lo),qstr(B.hi)],
        "y0_center":qstr(y0),"order":LOCAL_ORDER,"h":qstr(LOCAL_H),
        "cells":LOCAL_CELLS,"q":qstr(q),"V":qstr(V),"f_error":qstr(ferr),
        "D_lower":qstr(floor_decimal(narrow["Dmin"],40)),
        "polynomial_sha256":polynomial_hash(F,Y),
    }
    print(f"{label}: local q={float(q):.6f}, V={float(V):.3e}, D>={float(narrow['Dmin']):.12f}")
    return state,narrow["Dmin"],record


def validate_step(x0:Q,h:Q,state:tuple[IV,IV]):
    fc=round_decimal(state[0].midpoint(),STATE_PLACES)
    yc=round_decimal(state[1].midpoint(),STATE_PLACES)
    E0=max((state[0]-fc).abs_upper(),(state[1]-yc).abs_upper())
    F,Y=regular_taylor(x0,fc,yc)
    path=path_bounds(F,Y,x0,h,REGULAR_TRIAL,REGULAR_TRIAL,REGULAR_CELLS,True)
    residual=residual_polynomials(F,Y,x0,True)
    num=pl1(residual,h)
    den=9*x0*(2-(x0+h))*path["Dmin"]
    rho=ceil_decimal(num/den)
    L=ceil_decimal(path["L"],35)
    beta=ceil_decimal(exp_upper(L*h)*(E0+rho*h))
    if beta>=REGULAR_TRIAL:
        raise ArithmeticError(f"beta fails at {x0}: {beta}")
    state_out=(peval(F,h)+sym(beta),peval(Y,h)+sym(beta))
    return state_out,path["Dmin"],{
        "x0":qstr(x0),"h":qstr(h),"x1":qstr(x0+h),
        "E0":qstr(ceil_decimal(E0)),"rho":qstr(rho),"L":qstr(L),
        "beta":qstr(beta),"D_lower":qstr(floor_decimal(path["Dmin"],40)),
        "polynomial_sha256":polynomial_hash(F,Y),
    }


def run(Alo:Q,Ahi:Q,label:str):
    state,Dmin,local=validate_local(Alo,Ahi,label)
    x=LOCAL_H; steps=[]; i=0
    while x<1:
        h=min(x/Q(3),MAX_STEP,1-x)
        state,Dstep,rec=validate_step(x,h,state)
        Dmin=min(Dmin,Dstep); steps.append(rec); x+=h; i+=1
        if i%40==0 or x==1:
            print(f"{label}: step {i:3d}, x={float(x):.6f}, beta={float(Q(rec['beta'])):.3e}, Dmin={float(Dmin):.12f}")
    out={
        "label":label,"steps":i,"endpoint_f":[qstr(state[0].lo),qstr(state[0].hi)],
        "endpoint_y":[qstr(state[1].lo),qstr(state[1].hi)],
        "D_lower":qstr(floor_decimal(Dmin,40)),"local":local,"step_records":steps,
    }
    return out


def main() -> None:
    start=time.time()
    print(f"Python {platform.python_version()}; exact Fraction arithmetic; no third-party interval library")
    print(f"pi width <= {float(PI.hi-PI.lo):.3e}")
    runs={}
    runs["A_minus"]=run(A_MINUS,A_MINUS,"A_minus")
    runs["A_plus"]=run(A_PLUS,A_PLUS,"A_plus")
    fm=IV(Q(runs['A_minus']['endpoint_f'][0]),Q(runs['A_minus']['endpoint_f'][1]))
    fp=IV(Q(runs['A_plus']['endpoint_f'][0]),Q(runs['A_plus']['endpoint_f'][1]))
    if fm.lo<=0: raise ArithmeticError(f"A_minus sign not positive: {fm}")
    if fp.hi>=0: raise ArithmeticError(f"A_plus sign not negative: {fp}")
    dmin=min(Q(r['D_lower']) for r in runs.values())
    result={
        "schema":"special-lagrangian-independent-rational-validation-v1",
        "implementation":{
            "language":"CPython standard library only","arithmetic":"exact fractions and rational intervals",
            "python":platform.python_version(),"local_h":qstr(LOCAL_H),"local_order":LOCAL_ORDER,
            "local_cells":LOCAL_CELLS,"regular_order":REGULAR_ORDER,"regular_cells":REGULAR_CELLS,
            "regular_step_rule":"h=min(x/3,1/100,1-x)","regular_trial_radius":qstr(REGULAR_TRIAL),
            "coefficient_decimal_places":COEFF_PLACES,"state_decimal_places":STATE_PLACES,
            "transcendental_series_terms":TRANS_TERMS,
        },
        "runs":runs,
        "certified_claims":{
            "A_minus_positive":True,"A_plus_negative":True,
            "A_minus_f_at_s0":[qstr(fm.lo),qstr(fm.hi)],
            "A_plus_f_at_s0":[qstr(fp.lo),qstr(fp.hi)],
            "point_run_D_lower":qstr(dmin),
        },
        "elapsed_seconds":time.time()-start,
    }
    out_path=sys.argv[1] if len(sys.argv)>1 else "independent_rational_certificate.json"
    with open(out_path,"w",encoding="utf-8") as f: json.dump(result,f,indent=2,sort_keys=True)
    print(f"A_minus f(0) in [{float(fm.lo):.18e}, {float(fm.hi):.18e}]")
    print(f"A_plus  f(0) in [{float(fp.lo):.18e}, {float(fp.hi):.18e}]")
    print(f"point-run D >= {float(dmin):.15f}")
    print(f"wrote {out_path}")
    print("INDEPENDENT RATIONAL VALIDATION VERIFIED")


if __name__=="__main__":
    main()
