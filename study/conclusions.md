# OVO — Estudio de Fusión: Conclusiones

*Dataset Replica (8 escenas) · fusion_method=clip · GTJump determinista · commit 5fb61fe · 2026-06-14*

## Baselines

- **light**: mIoU=0.2663, AP_agn=0.1540, Num_Instances=68
- **aggressive**: mIoU=0.2280, AP_agn=0.0950, Num_Instances=83

## Hallazgos

- **[RQ1]** overlap(nuevo) vs overlap_old en light: ΔmIoU medio=0.0013  _(confianza: media)_
- **[RQ1]** overlap(nuevo) vs overlap_old en aggressive: ΔmIoU medio=-0.0002  _(confianza: media)_
- **[RQ1]** activar cooccurrence en jump drift (light) NO degrada: ΔmIoU=0.0006 (neutro) pero ΔAP_agnostic=0.0490 (+31.8%) — sube pureza de instancia, contradice la hipótesis a priori del playbook  _(confianza: alta)_
- **[RQ1]** activar cooccurrence en jump drift (aggressive) NO degrada: ΔmIoU=0.0040 (neutro) pero ΔAP_agnostic=0.0090 (+9.5%) — sube pureza de instancia, contradice la hipótesis a priori del playbook  _(confianza: alta)_
- **[RQ1]** cadena ganadora (media combinada): abl-cooccurrence-aabb-cossim-overlap  _(confianza: media)_
- **[RQ3]** parámetros relevantes (Morris mu*): ['cooccurrence_veto_threshold', 'th_cossim', 'th_points', 'th_overlap_ratio', 'th_overlap_cos', 'th_overlap_ratio_low']  _(confianza: media)_
- **[RQ2]** θ* (knee) light: mIoU=0.2745 AP_agn=0.2230  _(confianza: media)_
- **[RQ2]** θ* (knee) aggressive: mIoU=0.2353 AP_agn=0.1120  _(confianza: media)_
- **[RQ4]** consistencia entre condiciones: True  _(confianza: media)_
- **[VALIDACION-EXTERNA]** El hallazgo NO transfiere a ORB-SLAM real: el mecanismo cooccurrence que mejoraba AP_agnostic en jump-drift sintético lo DEGRADA bajo drift real. En ORB-SLAM real, el mecanismo cooccurrence cambia AP_agnostic en -0.0286 (-15.6%, Wilcoxon p=0.102, n=8) y mIoU en -0.0047 (-1.6%). NO transfiere el hallazgo del estudio sim.  _(confianza: alta)_

## Mejor configuración

- **light** θ*: `{"fusion_criteria": ["cooccurrence", "aabb", "cos_sim", "overlap"], "cooccurrence_veto_threshold": 9, "th_cossim": 0.7009442786285183, "th_points": 0.02711735548834665, "th_overlap_ratio": 0.3738508717482608, "th_overlap_cos": 0.927322451389549, "th_overlap_ratio_low": 0.27220801836397585}` → mIoU=0.2745, AP_agn=0.2230
- **aggressive** θ*: `{"fusion_criteria": ["cooccurrence", "aabb", "cos_sim", "overlap"], "cooccurrence_veto_threshold": 3, "th_cossim": 0.580146338133027, "th_points": 0.03821444603587826, "th_overlap_ratio": 0.6844640930984376, "th_overlap_cos": 0.9276162969382312, "th_overlap_ratio_low": 0.1394220566903776}` → mIoU=0.2353, AP_agn=0.1120
- **Recomendada (global)**: `{"fusion_criteria": ["cooccurrence", "aabb", "cos_sim", "overlap"], "cooccurrence_veto_threshold": 3, "th_cossim": 0.580146338133027, "th_points": 0.03821444603587826, "th_overlap_ratio": 0.6844640930984376, "th_overlap_cos": 0.9276162969382312, "th_overlap_ratio_low": 0.1394220566903776}`
  - knee de 'aggressive' generaliza mejor (menor caída relativa al cruzar de condición)

## Robustez

- consistent_across_conditions: 1.0000
- light_mIoU_std_across_scenes: 0.0896
- aggressive_mIoU_std_across_scenes: 0.0812
- light_loso_std_mIoU: 0.0128
- aggressive_loso_std_mIoU: 0.0116

## Limitaciones

- VALIDACIÓN EXTERNA: el óptimo sim NO transfiere a ORB-SLAM real (cooccurrence degrada AP_agnostic ~-15%); baseline gana en SLAM real
- solo fusion_method=clip
- óptimo condicionado a jump_seed=42
- backend SimulatedSLAM/GTJump determinista
- covisibilidad NO evaluada: no implementada en el código actual
- Fase 5 (Sobol) omitida por coste; interacción vía Morris σ
- presupuestos Optuna/Morris reducidos por coste de replay (~6 min/config)