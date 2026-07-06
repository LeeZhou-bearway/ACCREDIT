"""Recompute QCS v2 scores from existing JSON files (no re-warp needed).

Reads raw metrics (td, bd, hd, ir, nmi, cov, bg) from qcs_{px}px.json,
applies QCS v2 formula (TD=0.30 BD=0.30 HD=0.15 IR=0.10 NMI=0.15),
writes updated qcs field back to JSON + CSV.

Usage:
  python recompute_qcs_v2_from_existing.py
"""
import json, csv

OUT = '/home/zhoule/workspace/ACCREDIT/paper_figures/qcs_final_reference'

HD_MAX_REF = 500.0
IR_MAX = 0.15
NMI_MIN = 1.0
NMI_RANGE = 0.15
COV_MIN = 0.20

ALL_KEYS = [
    'Breast_Cancer', 'Cervical_Cancer', 'Lung_Cancer_Prime', 'Ovary',
    'Mouse_Brain', 'BoneMarrow', 'Liver', 'Pancreas_Cancer',
    'Colorectal_Cancer', 'Kidney', 'Pancreas', 'Skin', 'Heart',
    'humanLung', 'Prostate', 'Rep1', 'Rep2',
]


def qcs_v2(m, eval_px):
    hd_max = HD_MAX_REF * eval_px / 4000.0
    td = m.get('tissue_dice', 0) or 0
    bd = m.get('boundary_dice', 0) or 0
    hd = m.get('hausdorff_mean', hd_max) or hd_max
    ir = m.get('inlier_ratio', 0) or 0
    nmi = m.get('nmi', NMI_MIN) or NMI_MIN
    cov = m.get('coverage', 1.0) or 1.0
    bg = m.get('bg_coverage', 0) or 0

    s_td = max(0, min(1, td))
    s_bd = max(0, min(1, bd))
    s_hd = max(0, min(1, 1 - hd / hd_max))
    s_ir = max(0, min(1, ir / IR_MAX))
    s_nmi = max(0, min(1, (nmi - NMI_MIN) / NMI_RANGE))

    qcs = 0.30*s_td + 0.30*s_bd + 0.15*s_hd + 0.10*s_ir + 0.15*s_nmi
    qcs *= min(1, cov / COV_MIN) * max(0, 1 - bg)
    return round(qcs, 4)


def main():
    for px in [2000, 4000, 8000]:
        m = json.load(open(f'{OUT}/qcs_{px}px.json'))
        for k in m:
            for method in ['ours', 'valis']:
                if m[k].get(method):
                    m[k][method]['qcs_v1'] = m[k][method].get('qcs_v1', m[k][method].get('qcs'))
                    m[k][method]['qcs'] = qcs_v2(m[k][method], px)

        with open(f'{OUT}/qcs_{px}px.json', 'w') as f:
            json.dump(m, f, indent=2)

        with open(f'{OUT}/qcs_{px}px.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample', 'ours_qcs', 'valis_qcs', 'diff',
                        'ours_TD', 'valis_TD', 'ours_BD', 'valis_BD',
                        'ours_HD', 'valis_HD', 'ours_NMI', 'valis_NMI',
                        'ours_NCC', 'valis_NCC', 'ours_IR', 'valis_IR',
                        'ours_cov', 'valis_cov'])
            for k in ALL_KEYS:
                r = m.get(k, {})
                o = r.get('ours'); v = r.get('valis')
                if o and v:
                    w.writerow([k, o['qcs'], v['qcs'], round(o['qcs']-v['qcs'], 4),
                                o['tissue_dice'], v['tissue_dice'],
                                o['boundary_dice'], v['boundary_dice'],
                                o['hausdorff_mean'], v['hausdorff_mean'],
                                o.get('nmi', ''), v.get('nmi', ''),
                                o['ncc'], v['ncc'],
                                o['inlier_ratio'], v['inlier_ratio'],
                                o['coverage'], v['coverage']])
                else:
                    w.writerow([k] + ['N/A'] * 17)

        print(f'{px}px: updated QCS v2')
    print('Done')


if __name__ == '__main__':
    main()
