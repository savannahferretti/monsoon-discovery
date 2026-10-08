#!/usr/bin/env python

import os
import re
import ast
import json
import numpy as np
import pandas as pd

def get_operators():
    '''
    Purpose: NumPy versions of the operators used in PySR equations and in the hand-specified forms.
    Returns:
    - dict[str,callable]: operator name → function
    '''
    return {
        'cube':lambda x:x**3,
        'square':lambda x:x**2,
        'neg':lambda x:-x,
        'sqrt':np.sqrt,
        'exp':np.exp,
        'log':np.log,
        'abs':np.abs,
        'sin':np.sin,
        'cos':np.cos,
        'max':np.maximum,
        'min':np.minimum,
        'safepow':lambda a,b:np.abs(a)**b}

def prepare_form(form):
    '''
    Purpose: Rewrite PySR's `a^b` as safepow(a,b) so equation strings evaluate in Python.
    Args:
    - form (str): equation string
    Returns:
    - str: evaluable equation string
    '''
    return re.sub(r'(\w+)\^(\w+)',r'safepow(\1,\2)',form)

def extract_constants(form,predictornames):
    '''
    Purpose: List the constants in a form, i.e. names that are neither predictors nor operators.
    Args:
    - form (str): equation string
    - predictornames (list[str]): predictor names
    Returns:
    - list[str]: sorted constant names
    '''
    names = {node.id for node in ast.walk(ast.parse(form,mode='eval')) if isinstance(node,ast.Name)}
    return sorted(names-set(predictornames)-set(get_operators())-{'True','False','None'})

def evaluate(form,columns,constants):
    '''
    Purpose: Evaluate an equation on standardized predictors in float64. Every script evaluates equations here.
    Args:
    - form (str): equation string
    - columns (dict[str,np.ndarray]): standardized predictors; 'timeidx' is ignored
    - constants (dict[str,float]): constant values
    Returns:
    - np.ndarray: equation output z
    '''
    namespace = dict(get_operators(),__builtins__={})
    namespace.update({name:np.asarray(values,dtype=np.float64) for name,values in columns.items() if name!='timeidx'})
    namespace.update({name:float(value) for name,value in constants.items()})
    out = np.asarray(eval(prepare_form(form),namespace),dtype=np.float64)
    if out.ndim==0:
        out = np.full(len(next(values for name,values in columns.items() if name!='timeidx')),float(out))
    return out

def raw_to_precip(raw,std):
    '''
    Purpose: Convert equation output to precipitation, P = exp(s_y·max(z, 0)) − 1.
    Args:
    - raw (np.ndarray): equation output z
    - std (float): training standard deviation of log1p(precipitation)
    Returns:
    - np.ndarray: precipitation (mm)
    '''
    return np.expm1(std*np.maximum(raw,0.0))

def round_constants(constants,sigfigs):
    '''
    Purpose: Round constants to a number of significant figures.
    Args:
    - constants (dict[str,float]): constants
    - sigfigs (int): significant figures
    Returns:
    - dict[str,float]: rounded constants
    '''
    return {name:float(f'{value:.{sigfigs}g}') for name,value in constants.items()}

def load_registry(modelsdir):
    '''
    Purpose: Load the optimized-equations registry from CSV.
    Args:
    - modelsdir (str): models directory
    Returns:
    - dict[str,dict]: name → {form, constants, train_loss, valid_loss}
    '''
    filepath = os.path.join(modelsdir,'sr','optimized_equations.csv')
    if not os.path.exists(filepath):
        return {}
    return {row['name']:dict(form=row['form'],constants=json.loads(row['constants']),train_loss=row['train_loss'],valid_loss=row['valid_loss'])
            for _,row in pd.read_csv(filepath).iterrows()}

def save_registry(registry,modelsdir):
    '''
    Purpose: Save the optimized-equations registry as CSV and verify by reopening.
    Args:
    - registry (dict[str,dict]): name → {form, constants, train_loss, valid_loss}
    - modelsdir (str): models directory
    '''
    filepath = os.path.join(modelsdir,'sr','optimized_equations.csv')
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    rows = [dict(name=name,form=entry['form'],train_loss=entry['train_loss'],valid_loss=entry['valid_loss'],constants=json.dumps(entry['constants']))
            for name,entry in registry.items()]
    pd.DataFrame(rows).to_csv(filepath,index=False)
    if set(load_registry(modelsdir))!=set(registry):
        raise ValueError(f'{filepath} does not match the registry that was saved')

def calc_physical_constants(name,registry,stats):
    '''
    Purpose: Convert an optimized equation's standardized constants to its physical-space constants.
    Args:
    - name (str): equation name
    - registry (dict[str,dict]): optimized equations
    - stats (dict[str,float]): training statistics
    Returns:
    - dict[str,float]: physical constants
    '''
    sy   = stats['tp_std']
    mean = lambda var:stats[f'{var}_mean']
    std  = lambda var:stats[f'{var}_std']
    if name=='sr_bl_eq':
        c = registry['sr_bl_eq']['constants']
        return {'lam':sy/std('bl')**3,'bc':mean('bl')-c['c1']*std('bl'),'beta':sy*c['c2']}
    atm   = registry['sr_atm_eq']['constants']
    gamma = atm['c4']*std('thetae')/std('thetaestar')
    physical = {
        'lam':sy*atm['c3']/std('thetae')**3,
        'kappa':std('thetae')/std('rh'),
        'gamma':gamma,
        'thetac':mean('thetae')-gamma*mean('thetaestar')+atm['c5']*std('thetae')}
    if name=='sr_atm_eq':
        return physical
    c = registry[name]['constants']
    if name=='sr_sfc_eq':
        physical.update(lamshf=sy/std('shf'),lfc=c['c6'],lamlhf=sy*c['c7']/std('lhf'))
    elif name=='sr_all_eq':
        physical.update(lamthetae=sy*c['c8']/std('thetae'),lamshf=sy/std('shf'),lfc=c['c9'])
    else:
        raise ValueError(f'No physical form for `{name}`')
    return physical

def get_physical_form(name):
    '''
    Purpose: LaTeX physical-space form of an optimized equation's s_y·z, matching calc_physical_precip(). E_ATM is SR-ATM's s_y·z.
    Args:
    - name (str): equation name
    Returns:
    - str: LaTeX equation
    '''
    atm   = r'\lambda\max\left(\kappa(\mathrm{RH}-\mu_\mathrm{RH}),\,\theta_e-\gamma\theta_e^*-\Theta_c\right)^3'
    exponents = {
        'sr_bl_eq':r'\lambda(B_L-B_c)^3+\beta',
        'sr_atm_eq':atm,
        'sr_sfc_eq':r'E_\mathrm{ATM}+\lambda_\mathrm{SHF}(\mathrm{LF}_c-\mathrm{LF})(\mathrm{SHF}-\mu_\mathrm{SHF})+\lambda_\mathrm{LHF}(\mathrm{LHF}-\mu_\mathrm{LHF})',
        'sr_all_eq':r'E_\mathrm{ATM}+(\mathrm{LF}_c-\mathrm{LF})^3\left[\lambda_\mathrm{SHF}(\mathrm{SHF}-\mu_\mathrm{SHF})+\lambda_{\theta_e}(\theta_e-\mu_{\theta_e})\right]'}
    if name not in exponents:
        raise ValueError(f'No physical form for `{name}`')
    return f'${exponents[name]}$'

def calc_physical_precip(name,physical,inputs,stats):
    '''
    Purpose: Predict precipitation with the physical-space form of an optimized equation, P = max(exp(E) − 1, 0).
    Args:
    - name (str): equation name
    - physical (dict[str,float]): physical constants from calc_physical_constants()
    - inputs (dict[str,np.ndarray]): physical predictors (kernel-integrated rh, thetae, thetaestar, and bl, lf, shf, lhf)
    - stats (dict[str,float]): training statistics
    Returns:
    - np.ndarray: precipitation (mm)
    '''
    mean = lambda var:stats[f'{var}_mean']
    p    = physical
    if name=='sr_bl_eq':
        exponent = p['lam']*(inputs['bl']-p['bc'])**3+p['beta']
    else:
        moisture    = p['kappa']*(inputs['rh']-mean('rh'))
        instability = inputs['thetae']-p['gamma']*inputs['thetaestar']-p['thetac']
        exponent    = p['lam']*np.maximum(moisture,instability)**3
        if name=='sr_sfc_eq':
            exponent = exponent+p['lamshf']*(p['lfc']-inputs['lf'])*(inputs['shf']-mean('shf'))+p['lamlhf']*(inputs['lhf']-mean('lhf'))
        elif name=='sr_all_eq':
            exponent = exponent+(p['lfc']-inputs['lf'])**3*(p['lamshf']*(inputs['shf']-mean('shf'))+p['lamthetae']*(inputs['thetae']-mean('thetae')))
    return np.maximum(np.expm1(exponent),0.0)
