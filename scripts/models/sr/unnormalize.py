#!/usr/bin/env python
'''
Compute physical-space named constants for each optimized SR equation and report
Gaussian kernel parameters from the nn_gauss checkpoints.

Run on the cluster where stats.json, optimized_equations.pkl, and nn model
checkpoints are available.

Outputs a formatted table suitable for the paper appendix.
'''

import os
import sys
import json
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from scripts.utils import Config
from scripts.models.nn.classes.factory import build_model


def load_stats(config):
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'splits', 'stats.json'))
    with open(statsfile, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_registry(config):
    registrypath = os.path.join(config.modelsdir, 'sr', 'optimized_equations.pkl')
    with open(registrypath, 'rb') as f:
        return pickle.load(f)


def extract_kernel_params(config):
    '''
    Load each nn_gauss checkpoint, extract Gaussian kernel mu and logstd, and
    convert to sigma-level coordinates.

    The kernel coord = linspace(-1, 1, nlevs=11) maps to sigma-levels [0.50 .. 1.00]:
        sigma = 0.75 + coord * 0.25
    so:
        sigma_center = 0.75 + mu * 0.25
        sigma_width  = exp(logstd) * 0.25
    '''
    nn = config.nn
    runconfig = nn['runs']['nn_gauss']
    fieldvars = runconfig['fieldvars']
    seeds = nn['seeds']

    # build a dummy model to determine nlevs from a saved checkpoint
    modeldir = os.path.join(config.modelsdir, 'nn')
    results = []

    for seed in seeds:
        ckpt_path = os.path.join(modeldir, f'nn_gauss_{seed}.pth')
        if not os.path.exists(ckpt_path):
            print(f'  [warn] checkpoint not found: {ckpt_path}')
            continue
        state = torch.load(ckpt_path, map_location='cpu')
        # extract mu and logstd from state dict
        mu_key     = 'kernel.function.mu'
        logstd_key = 'kernel.function.logstd'
        if mu_key not in state:
            # try nested under 'model' or 'state_dict'
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model_state_dict' in state:
                state = state['model_state_dict']
        mu     = state[mu_key].numpy()      # shape (nfieldvars,)
        logstd = state[logstd_key].numpy()  # shape (nfieldvars,)
        std    = np.exp(logstd)

        sigma_center = 0.75 + mu * 0.25
        sigma_width  = std * 0.25

        results.append({'seed': seed, 'sigma_center': sigma_center, 'sigma_width': sigma_width})

    if not results:
        return None, fieldvars

    centers = np.stack([r['sigma_center'] for r in results], axis=0)  # (nseeds, nfieldvars)
    widths  = np.stack([r['sigma_width']  for r in results], axis=0)

    summary = {
        'fieldvars':     fieldvars,
        'center_mean':   centers.mean(axis=0),
        'center_std':    centers.std(axis=0),
        'width_mean':    widths.mean(axis=0),
        'width_std':     widths.std(axis=0),
    }
    return summary, fieldvars


def unnormalize_sr_bl(constants, stats, targetvar='tp'):
    '''
    Form: a * cube(bl + b) + c

    Physical-space (log1p(mm)) expansion:
        a * ((BL - mu_BL)/sigma_BL + b)^3 + c
        = (a / sigma_BL^3) * (BL - BL_crit)^3 + c

    In log1p(mm): multiply scale and intercept by sigma_tp and add mu_tp.
    '''
    a, b, c   = constants['a'], constants['b'], constants['c']
    mu_bl     = stats['bl_mean']
    sigma_bl  = stats['bl_std']
    sigma_tp  = stats[f'{targetvar}_std']
    mu_tp     = stats[f'{targetvar}_mean']

    BL_crit  = mu_bl - b * sigma_bl
    alpha_BL = sigma_tp * a / sigma_bl**3
    beta_BL  = sigma_tp * c + mu_tp

    return [
        ('BL_crit', BL_crit,  'm/s²',       'Critical BL for precipitation onset'),
        ('α_BL',    alpha_BL, '(m/s²)⁻³',   'Cubic sensitivity coefficient'),
        ('β_BL',    beta_BL,  '—',           'Intercept'),
    ]


def unnormalize_sr_lo(constants, stats, targetvar='tp'):
    '''
    Form: a * exp(b * rh) + c

    Physical-space expansion (rh is kernel-integrated column RH in %):
        a * exp(b * (RH - mu_RH)/sigma_RH) + c
        = a*exp(-b*mu_RH/sigma_RH) * exp((b/sigma_RH)*RH) + c
    '''
    a, b, c   = constants['a'], constants['b'], constants['c']
    mu_rh     = stats['rh_mean']
    sigma_rh  = stats['rh_std']
    sigma_tp  = stats[f'{targetvar}_std']
    mu_tp     = stats[f'{targetvar}_mean']

    alpha_LO = sigma_tp * a * np.exp(-b * mu_rh / sigma_rh)
    B_LO     = b / sigma_rh
    beta_LO  = sigma_tp * c + mu_tp

    return [
        ('α_LO', alpha_LO, '—',    'Exponential prefactor'),
        ('B_LO', B_LO,     '1/%',  'Exponential RH sensitivity'),
        ('β_LO', beta_LO,  '—',    'Intercept'),
    ]


def unnormalize_sr_med(constants, stats, targetvar='tp'):
    '''
    Form: a * cube(max(rh, thetae + b * thetaestar + c))

    The inner expression thetae_std + b*thetaestar_std + c expands to:
        (1/sigma_Θe) * (Θe - κ*Θe* - Θe_crit)
    where:
        κ        = -b * sigma_Θe / sigma_Θe*     (dimensionless; b < 0 so κ > 0)
        Θe_crit  = mu_Θe + b*(sigma_Θe/sigma_Θe*)*mu_Θe* - c*sigma_Θe   (K)

    Full equation in log1p(mm):
        α_MED * cube(max(rh_std, (Θe - κ*Θe* - Θe_crit)/sigma_Θe))

    To express the argument to cube in K, define RH_eff = (sigma_Θe/sigma_rh)*(RH - mu_rh):
        α_MED * cube(max(RH_eff, Θe - κ*Θe* - Θe_crit))
    where α_MED has units log1p(mm)/K^3.

    Note: rh_std*sigma_Θe = RH_eff is rh rescaled to the temperature scale, not physical %.
    '''
    a, b, c        = constants['a'], constants['b'], constants['c']
    mu_rh          = stats['rh_mean']
    sigma_rh       = stats['rh_std']
    mu_te          = stats['thetae_mean']
    sigma_te       = stats['thetae_std']
    mu_tes         = stats['thetaestar_mean']
    sigma_tes      = stats['thetaestar_std']
    sigma_tp       = stats[f'{targetvar}_std']

    kappa    = -b * sigma_te / sigma_tes
    Theta_crit = mu_te + b * (sigma_te / sigma_tes) * mu_tes - c * sigma_te
    alpha_MED  = sigma_tp * a / sigma_te**3
    rh_scale   = sigma_te / sigma_rh   # multiply by (RH - mu_rh) to get RH_eff in K

    return [
        ('α_MED',      alpha_MED,   'K⁻³',  'Cubic sensitivity coefficient'),
        ('κ',          kappa,       '—',     'θe* weighting relative to θe  (≈ 1 → CAPE proxy)'),
        ('Θe_crit',    Theta_crit,  'K',     'Critical buoyancy deficit for onset'),
        ('σ_Θe/σ_RH',  rh_scale,   'K/%',   'Scale factor: RH_eff [K] = (σ_Θe/σ_RH)·(RH − μ_RH)'),
    ]


def unnormalize_sr_hi(constants, stats, targetvar='tp'):
    '''
    Form: (a - b * lhf) * cube(max(rh, thetae + c * thetaestar + d))

    Flux factor:
        a - b*lhf_std = a - b*(LHF - mu_LHF)/sigma_LHF
                      = (a + b*mu_LHF/sigma_LHF) - (b/sigma_LHF)*LHF
                      = A_LHF - γ_LHF * LHF

    Cubic part: same as SR-MED with c → c, d → d (renaming from SR-MED's b,c).

    Full equation in log1p(mm):
        α_HI * (A_LHF - γ_LHF*LHF) * cube(max(RH_eff, Θe - κ*Θe* - Θe_crit))
    where α_HI = sigma_tp / sigma_Θe^3.
    '''
    a, b, c, d     = constants['a'], constants['b'], constants['c'], constants['d']
    mu_rh          = stats['rh_mean']
    sigma_rh       = stats['rh_std']
    mu_te          = stats['thetae_mean']
    sigma_te       = stats['thetae_std']
    mu_tes         = stats['thetaestar_mean']
    sigma_tes      = stats['thetaestar_std']
    mu_lhf         = stats['lhf_mean']
    sigma_lhf      = stats['lhf_std']
    sigma_tp       = stats[f'{targetvar}_std']

    A_LHF      = a + b * mu_lhf / sigma_lhf
    gamma_LHF  = b / sigma_lhf
    kappa      = -c * sigma_te / sigma_tes
    Theta_crit = mu_te + c * (sigma_te / sigma_tes) * mu_tes - d * sigma_te
    # alpha absorbs sigma_tp and 1/sigma_te^3; the (a - b*lhf_std) factor contributes its own
    # scale, so alpha_HI = sigma_tp * a / sigma_te^3 only when flux factor is normalized to 1.
    # For the fully expanded form: effective alpha = sigma_tp / sigma_te^3 (flux carries the a).
    alpha_HI   = sigma_tp / sigma_te**3
    rh_scale   = sigma_te / sigma_rh

    return [
        ('α_HI',       alpha_HI,   'K⁻³',   'Cubic sensitivity (flux-independent part)'),
        ('A_LHF',      A_LHF,      '—',      'LHF flux offset'),
        ('γ_LHF',      gamma_LHF,  'm²/W',   'LHF sensitivity of precipitation response'),
        ('κ',          kappa,       '—',      'θe* weighting relative to θe'),
        ('Θe_crit',    Theta_crit,  'K',      'Critical buoyancy deficit for onset'),
        ('σ_Θe/σ_RH',  rh_scale,   'K/%',    'Scale factor: RH_eff [K] = (σ_Θe/σ_RH)·(RH − μ_RH)'),
    ]


def roundtrip_check(name, entry, stats, targetvar='tp'):
    '''
    Verify that evaluating the physical-space equation at a test point gives the
    same result as the standardized equation + denormalization.
    '''
    from scripts.models.sr.optimize import eval_form, SRFUNCTIONS

    form    = entry['form']
    consts  = entry['constants']
    sigma_tp = stats[f'{targetvar}_std']
    mu_tp    = stats[f'{targetvar}_mean']

    # standardized test inputs (arbitrary but fixed)
    test_std = {'bl': 0.5, 'rh': 0.3, 'thetae': -0.2, 'thetaestar': 0.1,
                'lf': -0.4, 'shf': 0.6, 'lhf': -0.3}

    import pandas as pd
    x_df = pd.DataFrame([test_std])
    predictornames = list(set(test_std.keys()) & set(
        n.id for n in __import__('ast').walk(__import__('ast').parse(form, mode='eval'))
        if isinstance(n, __import__('ast').Name)))

    f_std = eval_form(form, x_df, list(x_df.columns), consts)
    if np.ndim(f_std) > 0:
        f_std = float(f_std[0])
    else:
        f_std = float(f_std)

    # standardized output → log1p(mm)
    f_lp_expected = float(sigma_tp * f_std + mu_tp)
    return f_lp_expected, f_std


def print_table(rows, title):
    col_widths = [max(len(str(r[i])) for r in rows + [(title, '', '', '')]) for i in range(4)]
    col_widths = [max(col_widths[0], 10), max(col_widths[1], 12), max(col_widths[2], 16), max(col_widths[3], 10)]
    fmt = '  {:<{}} {:>{}.4g} {:<{}} {}'
    sep = '  ' + '-' * (sum(col_widths) + 10)
    print(f'\n  {title}')
    print(sep)
    print(f'  {"Symbol":<{col_widths[0]}} {"Value":>{col_widths[1]}} {"Units":<{col_widths[2]}} Description')
    print(sep)
    for sym, val, units, desc in rows:
        print(fmt.format(sym, col_widths[0], val, col_widths[1], units, col_widths[2], desc))
    print(sep)


if __name__ == '__main__':
    config    = Config()
    targetvar = config.targetvar
    sr        = config.sr

    print('Loading stats.json ...')
    stats = load_stats(config)

    print('Loading optimized_equations.pkl ...')
    registry = load_registry(config)

    unnormalize_fns = {
        'sr_bl':  unnormalize_sr_bl,
        'sr_lo':  unnormalize_sr_lo,
        'sr_med': unnormalize_sr_med,
        'sr_hi':  unnormalize_sr_hi,
    }

    for name, entry in registry.items():
        fn = unnormalize_fns.get(name)
        if fn is None:
            print(f'\n[skip] No unnormalization function for {name}')
            continue
        rows = fn(entry['constants'], stats, targetvar=targetvar)
        desc = sr['optimizedeqs'].get(name, {}).get('description', name)
        title = f'{desc}  |  form: {entry["form"]}  |  valid_loss={entry["valid_loss"]:.4f}'
        print_table(rows, title)

    print('\nGaussian kernel parameters (nn_gauss, sigma-level coordinates)')
    print('  sigma-level: 0.50 (top) to 1.00 (surface), coord = linspace(-1,1,11)')
    kernel_summary, fieldvars = extract_kernel_params(config)
    if kernel_summary is not None:
        header = f'  {"Variable":<12} {"σ_center":>10} {"±":>4} {"σ_width":>10} {"±":>4}'
        sep    = '  ' + '-' * 46
        print(sep)
        print(header)
        print(sep)
        for i, fv in enumerate(kernel_summary['fieldvars']):
            print(f'  {fv:<12} {kernel_summary["center_mean"][i]:>10.4f} '
                  f'{kernel_summary["center_std"][i]:>4.4f} '
                  f'{kernel_summary["width_mean"][i]:>10.4f} '
                  f'{kernel_summary["width_std"][i]:>4.4f}')
        print(sep)
    else:
        print('  [warn] No kernel checkpoints found — run on the cluster.')
