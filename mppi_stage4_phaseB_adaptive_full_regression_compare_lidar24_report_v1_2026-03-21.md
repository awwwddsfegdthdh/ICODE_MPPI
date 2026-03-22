# 阶段4 PhaseB 固定 Seeds 完整回归对比（新：24束近距环形Lidar）

- 日期：2026-03-21
- 旧summary：`/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_phaseB_adaptive_full_regression_summary_v1_2026-03-21.json`
- 新summary：`/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_phaseB_adaptive_full_regression_summary_lidar24_v1_2026-03-21.json`
- 新输出目录：`/tmp/mppi_stage4_phaseB_adaptive_20260321_lidar24`
- fixed seeds：`6101,6102,6103,6104,6105`

## 1. 新旧阶段结果对比

| 配置 | final_dist_mean(old->new) | mppi_steps_mean(old->new) | override_steps_mean(old->new) | post_delta_ratio_mppi(old->new) | goal_blocked_conf_mean(old->new) |
|---|---:|---:|---:|---:|---:|
| r0_baseline_current | 0.952->1.055 | 187.8->182.0 | 52.2->58.0 | 0.1761->0.1890 | 0.00000->0.00000 |
| b1_a_conf_strict | 0.952->1.055 | 187.8->182.0 | 52.2->58.0 | 0.1761->0.1890 | 0.00000->0.00000 |
| b1_b_map_occ_up | 0.947->1.046 | 187.8->182.0 | 52.2->58.0 | 0.1761->0.1890 | 0.00000->0.00000 |
| b2_a_boundary_relax | 0.947->1.046 | 187.8->182.0 | 52.2->58.0 | 0.1761->0.1890 | 0.00000->0.00000 |
| b2_b_boundary_target_relax | 1.038->0.925 | 182.0->187.8 | 58.0->52.2 | 0.1956->0.1898 | 0.00000->0.00000 |
| b2_c_jam_relax1 | 0.644->0.618 | 211.0->211.0 | 29.0->29.0 | 0.1479->0.1498 | 0.00002->0.00000 |
| b2_d_jam_relax2 | 0.653->0.615 | 211.0->211.0 | 29.0->29.0 | 0.1479->0.1469 | 0.00002->0.00000 |

## 2. 失败链路差异（baseline 与 best）

- old_best：`b2_c_jam_relax1`
- new_best：`b2_d_jam_relax2`

### r0_baseline_current

| seed | first_fail_step old->new | trigger_around old->new | final_dist old->new | mppi/override old->new |
|---:|---:|---|---:|---:|
| 6101 | 17->17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 1.146->0.966 | 182/58->182/58 |
| 6102 | 17->17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.659->1.064 | 211/29->182/58 |
| 6103 | 17->17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.989->1.097 | 182/58->182/58 |
| 6104 | 17->17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.989->1.156 | 182/58->182/58 |
| 6105 | 17->17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.978->0.989 | 182/58->182/58 |

### b2_c_jam_relax1

| seed | first_fail_step old->new | trigger_around old->new | final_dist old->new | mppi/override old->new |
|---:|---:|---|---:|---:|
| 6101 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.571->0.565 | 211/29->211/29 |
| 6102 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.684->0.674 | 211/29->211/29 |
| 6103 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.565->0.470 | 211/29->211/29 |
| 6104 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.627->0.608 | 211/29->211/29 |
| 6105 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.775->0.772 | 211/29->211/29 |

### b2_d_jam_relax2

| seed | first_fail_step old->new | trigger_around old->new | final_dist old->new | mppi/override old->new |
|---:|---:|---|---:|---:|
| 6101 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.571->0.565 | 211/29->211/29 |
| 6102 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.684->0.674 | 211/29->211/29 |
| 6103 | 28->28 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.608->0.454 | 211/29->211/29 |
| 6104 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.627->0.608 | 211/29->211/29 |
| 6105 | 23->23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence']->['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.775->0.772 | 211/29->211/29 |
