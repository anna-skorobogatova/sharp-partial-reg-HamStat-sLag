#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

PKG = Path('/mnt/data/validation_archive/special_lagrangian_validation')
CERT = json.loads((PKG/'certificate.json').read_text())

def require(cond: bool, msg: str) -> None:
    if not cond: raise AssertionError(msg)

def frac(s: Any) -> Fraction:
    if isinstance(s, Fraction): return s
    if isinstance(s, int): return Fraction(s)
    t=str(s)
    if '/' in t:
        a,b=t.split('/',1); return Fraction(int(a), int(b))
    return Fraction(Decimal(t))

def scaled(i:int,e:int)->Fraction:
    return Fraction(i*10**e) if e>=0 else Fraction(i,10**(-e))

def is_ball(v:Any)->bool:
    return isinstance(v,dict) and {'mid_integer','radius_integer','exponent10','lower_rational','upper_rational'} <= set(v)

def balls(v:Any,path='certificate')->Iterable[tuple[str,dict]]:
    if is_ball(v): yield path,v; return
    if isinstance(v,dict):
        for k,c in v.items(): yield from balls(c,f'{path}.{k}')
    elif isinstance(v,list):
        for i,c in enumerate(v): yield from balls(c,f'{path}[{i}]')

def lo(r): return frac(r['lower_rational'])
def hi(r): return frac(r['upper_rational'])
def contains(outer,inner): return lo(outer)<=lo(inner) and hi(inner)<=hi(outer)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

count=0
for p,r in balls(CERT):
    m=int(r['mid_integer']); rad=int(r['radius_integer']); e=int(r['exponent10'])
    require(rad>=0,f'{p}: negative radius')
    require(lo(r)==scaled(m-rad,e),f'{p}: lower serialization mismatch')
    require(hi(r)==scaled(m+rad,e),f'{p}: upper serialization mismatch')
    require(lo(r)<=hi(r),f'{p}: reversed interval')
    count+=1
require(count==4074,f'unexpected ball count {count}')
print(f'PASS: exact-rational serialization of {count} interval balls')

env=json.loads((PKG/'environment.json').read_text())
for filename,expected in env.get('source_sha256',{}).items():
    require(sha(PKG/filename)==expected,f'hash mismatch: {filename}')
for key,filename in [('certificate_sha256','certificate.json'),('continuation_steps_csv_sha256','continuation_steps.csv'),('endpoint_enclosures_sha256','endpoint_enclosures.json')]:
    if env.get(key): require(sha(PKG/filename)==env[key],f'hash mismatch: {filename}')
expected=(PKG/'certificate.sha256').read_text().split()[0]
require(sha(PKG/'certificate.json')==expected,'certificate.sha256 mismatch')
print('PASS: archived source/data hashes and certificate checksum')

cfg=CERT['configuration']
require(cfg['precision_bits']==512,'precision mismatch')
require(cfg['local_order']==40 and cfg['regular_order']==30,'Taylor order mismatch')
require(cfg['regular_step_rule']=='h=min(x/4,max_regular_step,1-x)','step rule mismatch')
local_h=frac(cfg['local_h']); max_step=frac(cfg['max_regular_step'])
threshold=frac('4.2964512605')
for name,run in CERT['runs'].items():
    local=run['local_taylor_validation']
    require(local['order_y']==40 and local['order_f']==41,f'{name}: local order')
    require(hi(local['contraction_q'])<1,f'{name}: q>=1')
    require(hi(local['tail_y_V'])<frac(local['trial_radius']),f'{name}: local tail exceeds radius')
    require(lo(local['D_tube'])>0,f'{name}: local D nonpositive')
    steps=run['continuation_steps']
    require(len(steps)==run['continuation_step_count']==103,f'{name}: step count')
    x=local_h; global_D=run['D_global_tube']
    require(contains(global_D,local['D_tube']),f'{name}: global D does not contain local D')
    for idx,step in enumerate(steps):
        require(step['index']==idx,f'{name} step {idx}: index')
        require(frac(step['x0'])==x,f'{name} step {idx}: broken chain')
        h=min(x/4,max_step,1-x)
        require(frac(step['h'])==h,f'{name} step {idx}: step rule')
        require(frac(step['x1'])==x+h,f'{name} step {idx}: x1')
        require(step['order_y']==30 and step['order_f']==31,f'{name} step {idx}: order')
        require(lo(step['residual_denominator_lower'])>0,f'{name} step {idx}: denominator')
        require(lo(step['D_tube'])>0,f'{name} step {idx}: D')
        require(hi(step['propagated_error_beta'])<frac(step['trial_radius']),f'{name} step {idx}: beta')
        require(contains(global_D,step['D_tube']),f'{name} step {idx}: D hull')
        x+=h
    require(x==1,f'{name}: does not reach x=1')
    require(lo(run['endpoint_x1_s0']['f'])==lo(steps[-1]['state_out']['f']) and hi(run['endpoint_x1_s0']['f'])==hi(steps[-1]['state_out']['f']),f'{name}: endpoint f mismatch')
    require(lo(run['endpoint_x1_s0']['y'])==lo(steps[-1]['state_out']['y']) and hi(run['endpoint_x1_s0']['y'])==hi(steps[-1]['state_out']['y']),f'{name}: endpoint y mismatch')
    print(f'PASS: {name} contraction and {len(steps)}-step continuation chain')
require(lo(CERT['runs']['A_interval']['D_global_tube'])>threshold,'uniform D threshold failure')
require(lo(CERT['runs']['A_minus']['endpoint_x1_s0']['f'])>0,'A_minus endpoint not positive')
require(hi(CERT['runs']['A_plus']['endpoint_x1_s0']['f'])<0,'A_plus endpoint not negative')
print('PASS: strict endpoint signs and simultaneous uniform D lower bound')

with (PKG/'continuation_steps.csv').open(newline='') as f: rows=list(csv.DictReader(f))
require(len(rows)==309,'CSV row count')
by={(r['run'],int(r['index'])):r for r in rows}; require(len(by)==309,'duplicate CSV keys')
for name,run in CERT['runs'].items():
    for step in run['continuation_steps']:
        r=by[(name,step['index'])]
        require(r['x0']==step['x0'] and r['h']==step['h'],f'{name} step {step["index"]}: CSV coordinates')
        require(r['polynomial_sha256']==step['polynomial_sha256'],f'{name} step {step["index"]}: CSV hash')
        require(frac(r['D_tube_lower'])==lo(step['D_tube']),f'{name} step {step["index"]}: CSV D')
print('PASS: all 309 CSV continuation records match certificate')
print('EXACT CERTIFICATE CHECK VERIFIED')
