#!/usr/bin/env python

import os
import re
import ast
import json
import numpy as np
import pandas as pd

FUNCTIONS = {
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
    '_safepow':lambda a,b:np.abs(a)**b}

def prepare_form(form):
    '''
    Purpose: Replace `a^b` with a safe power so PySR strings evaluate in Python.
    Args:
    - form (str): equation string
    Returns:
    - str: evaluable equation string
    '''
    return re.sub(r'(\w+)\^(\w+)',r'_safepow(\1,\2)',form)

def extract_constants(form,predictornames):
    '''
    Purpose: Names in a form that are neither predictors nor functions.
    Args:
    - form (str): equation string
    - predictornames (list[str]): predictor names
    Returns:
    - list[str]: sorted constant names
    '''
    names = {node.id for node in ast.walk(ast.parse(form,mode='eval')) if isinstance(node,ast.Name)}
    return sorted(names-set(predictornames)-set(FUNCTIONS)-{'True','False','None'})

def evaluate(form,columns,constants):
    '''
    Purpose: Evaluate an equation in standardized space. This is the only equation evaluator in the timing test.
    Args:
    - form (str): equation string
    - columns (dict[str, np.ndarray]): standardized predictors (float64)
    - constants (dict[str, float]): constant values
    Returns:
    - np.ndarray: raw output (float64)
    '''
    namespace = dict(FUNCTIONS,__builtins__={})
    namespace.update({name:np.asarray(values,dtype=np.float64) for name,values in columns.items()})
    namespace.update({name:float(value) for name,value in constants.items()})
    out = np.asarray(eval(prepare_form(form),namespace),dtype=np.float64)
    if out.ndim==0:
        out = np.full(len(next(iter(columns.values()))),float(out))
    return out

def raw_to_precip(raw,stats):
    '''
    Purpose: Convert raw equation output to precipitation, P = exp(s_y·max(raw, 0)) − 1 (Equation 1, Text S1).
    Args:
    - raw (np.ndarray): raw equation output
    - stats (dict): training statistics
    Returns:
    - np.ndarray: precipitation (mm)
    '''
    return np.expm1(stats['tp_std']*np.maximum(raw,0.0))

def round_constants(constants,sigfigs):
    '''
    Purpose: Round constants to a number of significant figures.
    Args:
    - constants (dict[str, float]): constants
    - sigfigs (int): significant figures
    Returns:
    - dict[str, float]: rounded constants
    '''
    return {name:float(f'{value:.{sigfigs}g}') for name,value in constants.items()}

def load_registry(config):
    '''
    Purpose: Load optimized constants from {modelsdir}/sr/optimized_equations.csv.
    Args:
    - config (TimingConfig): configuration object
    Returns:
    - dict[str, dict]: name → {form, constants, train_loss, valid_loss}
    '''
    filepath = os.path.join(config.modelsdir,'sr','optimized_equations.csv')
    if not os.path.exists(filepath):
        return {}
    return {row['name']:dict(form=row['form'],constants=json.loads(row['constants']),train_loss=row['train_loss'],valid_loss=row['valid_loss'])
            for _,row in pd.read_csv(filepath).iterrows()}

def save_registry(registry,config):
    '''
    Purpose: Save optimized constants to {modelsdir}/sr/optimized_equations.csv.
    Args:
    - registry (dict): name → {form, constants, train_loss, valid_loss}
    - config (TimingConfig): configuration object
    '''
    filepath = os.path.join(config.modelsdir,'sr','optimized_equations.csv')
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    rows = [dict(name=name,form=entry['form'],train_loss=entry['train_loss'],valid_loss=entry['valid_loss'],constants=json.dumps(entry['constants']))
            for name,entry in registry.items()]
    pd.DataFrame(rows).to_csv(filepath,index=False)

def calc_physical_constants(name,registry,stats,atmname='sr_atm_eq'):
    '''
    Purpose: Physical-space constants of an equation, as in notebooks/equations.ipynb.
    Args:
    - name (str): equation name
    - registry (dict): optimized constants
    - stats (dict): training statistics
    - atmname (str): name of the variant's SR-ATM equation (SR-SFC and SR-ALL build on it)
    Returns:
    - dict[str, float]: physical constants
    '''
    sy   = stats['tp_std']
    mean = lambda var:stats[f'{var}_mean']
    std  = lambda var:stats[f'{var}_std']
    if name=='sr_bl_eq':
        c = registry['sr_bl_eq']['constants']
        return {'lam':sy/std('bl')**3,'bc':mean('bl')-c['c1']*std('bl'),'beta':sy*c['c2']}
    if name=='sr_bl_exp_eq':
        c = registry['sr_bl_exp_eq']['constants']
        return {'amp':sy,'k':c['c17']/std('bl'),'b0':mean('bl')+c['c18']*std('bl')}
    if atmname=='sr_atm_sum_eq':
        c      = registry['sr_atm_sum_eq']['constants']
        gammap = c['c19']*std('thetaestar')/std('thetae')
        physical = {'lamm':sy/std('rh')**3,'lami':sy/std('thetaestar')**3,'gammap':gammap,
                    'theta0':gammap*mean('thetae')-mean('thetaestar')}
    else:
        atm   = registry['sr_atm_eq']['constants']
        gamma = atm['c4']*std('thetae')/std('thetaestar')
        physical = {'lam':sy*atm['c3']/std('thetae')**3,'kappa':std('thetae')/std('rh'),'gamma':gamma,
                    'thetac':mean('thetae')-gamma*mean('thetaestar')+atm['c5']*std('thetae')}
    if name in ('sr_atm_eq','sr_atm_sum_eq'):
        return physical
    c = registry[name]['constants']
    if name=='sr_sfc_eq':
        physical.update(lamshf=sy*c['c6']/std('shf'),lfc=c['c7'],lamlhf=sy*c['c8']/std('lhf'))
    elif name=='sr_all_eq':
        physical.update(lamthetae=sy/std('thetae'),lamshf=sy*c['c9']/std('shf'),lfc=c['c10'])
    elif name=='sr_all_pc_eq':
        physical.update(lamthetae=sy/std('thetae'),lamshf=sy*c['c12']/std('shf'),lfc=c['c13'])
    elif name=='sr_all_k1_eq':
        physical.update(lamthetae=sy*c['c14']/std('thetae'),lamshf=sy*c['c14']*c['c15']/std('shf'),lfc=c['c16'])
    else:
        raise ValueError(f'No physical form for `{name}`')
    return physical

def calc_physical_precip(name,physical,inputs,stats):
    '''
    Purpose: Precipitation from the physical-space form of a manuscript equation (Equations in Section 3 and
        Table S2), P = max(exp(E) − 1, 0).
    Args:
    - name (str): equation name
    - physical (dict[str, float]): physical constants from calc_physical_constants()
    - inputs (dict[str, np.ndarray]): physical predictors (kernel-integrated RH, θe, θe*, and bl, shf, lhf, lf)
    - stats (dict): training statistics
    Returns:
    - np.ndarray: precipitation (mm)
    '''
    mean = lambda var:stats[f'{var}_mean']
    p    = physical
    if name=='sr_bl_eq':
        exponent = p['lam']*(inputs['bl']-p['bc'])**3+p['beta']
    elif name=='sr_bl_exp_eq':
        exponent = p['amp']*np.exp(p['k']*(inputs['bl']-p['b0']))
    else:
        if 'lamm' in p:
            exponent = p['lamm']*(inputs['rh']-mean('rh'))**3+p['lami']*(p['gammap']*inputs['thetae']-inputs['thetaestar']-p['theta0'])**3
        else:
            moisture    = p['kappa']*(inputs['rh']-mean('rh'))
            instability = inputs['thetae']-p['gamma']*inputs['thetaestar']-p['thetac']
            exponent    = p['lam']*np.maximum(moisture,instability)**3
        if name=='sr_sfc_eq':
            exponent = exponent+p['lamshf']*(p['lfc']-inputs['lf'])*(inputs['shf']-mean('shf'))+p['lamlhf']*(inputs['lhf']-mean('lhf'))
        elif name in ('sr_all_eq','sr_all_pc_eq','sr_all_k1_eq'):
            exponent = exponent+(p['lfc']-inputs['lf'])**3*(p['lamthetae']*(inputs['thetae']-mean('thetae'))+p['lamshf']*(inputs['shf']-mean('shf')))
            if name=='sr_all_pc_eq':
                exponent = exponent+p['lamthetae']*(1-p['lfc'])**3*(inputs['thetae']-mean('thetae'))
    return np.maximum(np.expm1(exponent),0.0)
