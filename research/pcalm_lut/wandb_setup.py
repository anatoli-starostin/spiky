"""W&B wiring for this study, following the repo's existing convention (spiky.util.wandb_integration).

Project / entity / base URL come from the environment as everywhere else in the repo (WANDB_PROJECT,
WANDB_ENTITY, WANDB_BASE_URL; credentials from ~/.netrc, never from code). The GROUP is fixed here, as the
repo's rule requires the group to be set by the code rather than an env var: `pcalm-lut-paired`.

Every metric this study logs is described in GLOSSARY -- the repo's binding rule (claude/wandb.md section 5)
is that a run's notes describe the experiment AND every metric it logs.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.util.wandb_integration.glossary import DictGlossary  # noqa: E402
from spiky.util.wandb_integration.tracker import Tracker  # noqa: E402

GROUP = 'pcalm-lut-paired'
PROJECT = os.environ.get('WANDB_PROJECT', 'Spiky')

_METRICS = {
    'train/loss': dict(unit='sq.err/sample', section='train',
                       desc='The arm objective on the batch: 1/2||yhat-y||^2 per sample for BP; the relaxed '
                            'energy E (arm A) or L_rho (arm B) per sample for the PC arms.'),
    'train/loss_at_h': dict(unit='sq.err/sample', section='train',
                            desc='1/2||readout(h*)-y||^2 per sample at the end of the inner loop (PC arms).'),
    'train/s_per_step': dict(unit='s/step', section='train', desc='Wall clock of one optimiser step.'),
    'train/rho': dict(unit='-', section='train', desc='ALM penalty rho, raised by the schedule when the '
                                                      'residual fails to contract (arm B).'),
    'train/eta_h': dict(unit='-', section='train',
                        desc='Inner-loop step size, 2/(sigma_max^2 (2 rho + alpha)).'),
    'train/sigma_max': dict(unit='-', section='train',
                            desc='Largest singular value of A = dr/dh, power iteration (A v by finite '
                                 'differences, A^T u by one backward); re-measured periodically.'),
    'train/resid_rms': dict(unit='-', section='train', desc='RMS of the forward residuals r^f at the end of '
                                                            'the inner loop.'),
    'train/r_contraction': dict(unit='ratio', section='train',
                                desc='||r|| at the last inner step divided by the LARGEST ||r|| reached during '
                                     'the loop: how much of the constraint excursion the relaxation walked '
                                     'back (a last/first ratio is 0/0 when h starts at the forward pass).'),
    'train/r_peak': dict(unit='-', section='train', desc='Largest ||r^f|| reached during the inner loop.'),
    'train/r_last': dict(unit='-', section='train', desc='||r^f|| at the end of the inner loop.'),
    'train/energy_monotone': dict(unit='0/1', section='train',
                                  desc='1 if the inner-loop energy was non-increasing at every inner step.'),
    'eval/test_acc': dict(unit='fraction', section='eval',
                          desc='Top-1 accuracy on the first 2,000 test images.'),
    'eval/train_acc': dict(unit='fraction', section='eval',
                           desc='Top-1 accuracy on a FIXED 2,000-row train subset (--train-eval only), so '
                                'that train-minus-test is a generalisation gap and not minibatch noise.'),
    'eval/gap': dict(unit='fraction', section='eval',
                     desc='eval/train_acc - eval/test_acc: the generalisation gap (--train-eval only).'),
    'eval/train_loss_full': dict(unit='sq.err/sample', section='eval',
                                 desc='1/2||yhat-y||^2 per sample on the same fixed 2,000-row train subset, '
                                      'measured with dropout OFF (--train-eval only).'),
    'norm/f_tables_L{i}': dict(unit='-', section='collapse',
                               desc='RMS table entry of forward LUT i. Arm A has a trivial solution -- every '
                                    'block the identity, all tables zero -- whose signature is a monotone '
                                    'decay of this toward zero.'),
    'norm/f_tables_mean': dict(unit='-', section='collapse',
                               desc='Mean over forward LUTs of the RMS table entry.'),
    'ident/branch_ratio_L{i}': dict(unit='ratio', section='collapse',
                                    desc='||a_i LUT_i(h_i)|| / ||h_i|| for interior block i: how far the '
                                         'block moves the residual stream. A block that has collapsed to '
                                         'the identity map reports ~0.'),
    'ident/branch_ratio_mean': dict(unit='ratio', section='collapse',
                                    desc='Mean over interior blocks of the residual-branch ratio.'),
    'flips/mean': dict(unit='fraction', section='addresses',
                       desc='Mean over layers of the fraction of (sample, table) address slots whose selected '
                            'cell differs from the forward-pass cell after the inner loop.'),
    'flips/L{i}': dict(unit='fraction', section='addresses', desc='Address-flip fraction for forward LUT i.'),
    'margin/m_min_p50_L{i}': dict(unit='-', section='margins',
                                  desc='Median over (sample, table) of the SMALLEST anchor margin m_j* in '
                                       'forward LUT i. The routing block of the Jacobian carries '
                                       '(2/tau) w0 w1 ~ exp(-2 m_j*/tau), so this is the conditioning '
                                       'diagnostic: rising margins kill address search.'),
    'margin/m_min_p10_L{i}': dict(unit='-', section='margins', desc='10th percentile of the same quantity.'),
    'margin/m_min_p90_L{i}': dict(unit='-', section='margins', desc='90th percentile of the same quantity.'),
    'margin/m_sum_p50_L{i}': dict(unit='-', section='margins',
                                  desc='Median of sum_j m_j per (sample, table) in forward LUT i; sets the '
                                       'scale of the confidence score s_t.'),
    'margin/m_min_p50_mean': dict(unit='-', section='margins',
                                  desc='Mean over layers of the median smallest margin.'),
    'tau/f_L{i}': dict(unit='-', section='tau', desc='Blend temperature tau = exp(log_tau) of forward LUT i '
                                                     '(trainable; larger tau widens the routing window).'),
    'tau/g_L{i}': dict(unit='-', section='tau', desc='Blend temperature of backward LUT i.'),
    'align/f_mean': dict(unit='cosine', section='alignment',
                         desc='Mean over forward-LUT parameters of the cosine between this arm gradient and '
                              'the BP gradient on the same weights and probe batch.'),
    'align/g_mean': dict(unit='cosine', section='alignment', desc='Same, over backward-LUT parameters.'),
    'align/dead_layers': dict(unit='count', section='alignment',
                              desc='Number of parameter tensors receiving EXACTLY zero gradient this step.'),
    'align/f_tables_L{i}': dict(unit='cosine', section='alignment',
                                desc='Cosine to the BP gradient for the tables of forward LUT i.'),
    'align/g_grad_frac': dict(unit='fraction', section='alignment',
                              desc='Fraction of backward-LUT (g) parameter tensors receiving a nonzero '
                                   'gradient this step. g has no BP reference gradient at all -- it is not on '
                                   'the forward path -- so coverage replaces cosine for g.'),
    'out/readout_rms': dict(unit='-', section='magnitude',
                            desc='RMS of what the readout actually emits on the probe batch.'),
    'out/target_rms': dict(unit='-', section='magnitude',
                           desc='RMS of the one-hot target, 1/sqrt(C): the scale the readout has to reach.'),
    'out/readout_over_target': dict(unit='ratio', section='magnitude',
                                    desc='out/readout_rms divided by out/target_rms. Near 0 means the '
                                         'readout is emitting nothing whatever its gradient direction is.'),
    'wall_s': dict(unit='s', section='summary', desc='Total wall clock of the run (summary).'),
    'final_loss': dict(unit='sq.err/sample', section='summary', desc='train/loss at the last step (summary).'),
    'test_acc': dict(unit='fraction', section='summary', desc='eval/test_acc at the last probe (summary).'),
}

# Gradient / applied-update magnitude by bucket. The buckets are enumerated rather than patterned because
# the glossary only expands {i} into an integer, and these keys carry names.
_WHERE = {'readout': 'the readout LUT', 'interior': 'the interior forward LUTs',
          'backward': 'the backward stack g', 'input': 'the input linear map'}
_KIND = {'tables': 'table entries', 'log_tau': 'the learnable read temperature',
         'anchors': 'the anchor indices', 'W1': 'the weight matrix', 'other': 'the remaining parameters'}
for _w, _wd in _WHERE.items():
    for _k, _kd in _KIND.items():
        _b = f'{_w}_{_k}'
        _METRICS[f'grad/{_b}_rms'] = dict(
            unit='-', section='magnitude',
            desc=f'RMS of the ARM\'s weight gradient over {_kd} of {_wd}, on the probe batch.')
        _METRICS[f'grad/{_b}_rms_bp'] = dict(
            unit='-', section='magnitude',
            desc=f'RMS of BACKPROP\'s gradient over the same parameters at the same weights.')
        _METRICS[f'grad/{_b}_ratio'] = dict(
            unit='ratio', section='magnitude',
            desc=f'grad/{_b}_rms divided by grad/{_b}_rms_bp: how much harder than backprop this arm '
                 f'drives {_kd} of {_wd}.')
        _METRICS[f'update/{_b}_rms'] = dict(
            unit='-', section='magnitude',
            desc=f'RMS of the parameter change the optimiser ACTUALLY applied to {_kd} of {_wd} '
                 f'(~lr under Adam whatever the gradient scale; lr*grad under plain SGD).')

GLOSSARY = DictGlossary(
   _METRICS,
   sections={'train': 'Training', 'eval': 'Evaluation', 'addresses': 'Address search', 'summary': 'Summary',
             'collapse': 'Collapse / identity diagnostics', 'magnitude': 'Gradient and update magnitude',
             'margins': 'Margin distribution (conditioning)', 'tau': 'Blend temperature',
             'alignment': 'Gradient alignment to BP'},
   source='research/pcalm_lut/wandb_setup.py')


def make_tracker(cfg, out_dir, *, name, tags=()):
    """Tracker for this study; off (with one printed line) when the environment is not configured."""
    return Tracker.start(cfg, out_dir, project=PROJECT, group=GROUP, name=name, tags=tuple(tags),
                         glossary=GLOSSARY, code_dir=os.path.expanduser('~/projects/spiky'))
