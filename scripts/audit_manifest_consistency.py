#!/usr/bin/env python3
"""Read-only *internal* manifest consistency test; does not validate external waveform/model files."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
def data(relative_path):
    return json.loads((ROOT / relative_path).read_text())

s1=data('config/step1_manifest.json')
s2=data('config/step2_manifest.json')
fm=data('config/final_model_manifest.json')
pa=data('config/manifests/ptbxl/ptbxl_audit_manifest.json')
pe=data('config/manifests/ptbxl/ptbxl_evaluation_manifest.json')
pb=data('config/manifests/ptbxl/ptbxl_patient_cluster_bootstrap_manifest.json')
rc=data('config/manifests/reviewer/reviewer_control_manifest.json')
rb=data('config/manifests/reviewer/ptbxl_reviewer_control_bootstrap_manifest.json')
la=data('config/manifests/ludb/ludb_audit_manifest.json')
lb=data('config/manifests/ludb/ludb_step5b_manifest.json')
lc=data('config/manifests/ludb/ludb_step5c_manifest.json')
ld=data('config/manifests/ludb/ludb_step5d_manifest.json')
historical=data('config/archive_exploratory/experiment_config.json')
combined=data('config/manifests/reviewer/combined_loss_manifest.json')
perm=data('config/manifests/reviewer/permutation_manifest.json')

checks = [
('frozen ECG input hash',len({s1['frozen_ecg_sha256'],s2['frozen_V9_sha256'],fm['frozen_V9_sha256']})==1),
('gaze target hash',len({s1['clean_gaze_targets_sha256'],s2['frozen_gaze_target_sha256'],fm['frozen_target_sha256']})==1),
('base model hash',len({s1['base_model_sha256'],s2['base_model_sha256'],fm['base_model_sha256'],pe['base_model_sha256'],rc['base_model_sha256'],lc['base_model_sha256']})==1),
('consultant/all weight hashes',fm['consultant_weights_sha256']==pe['consultant_weights_sha256']==lc['consultant_weights_sha256'] and fm['all_clinician_weights_sha256']==pe['all_clinician_weights_sha256']==lc['all_clinician_weights_sha256']),
('CV 5 x 8-train/2-test, seed 2026',s2['seed']==2026 and s2['n_folds']==5 and s2['training_ECGs_per_fold']==8 and s2['held_out_ECGs_per_fold']==2),
('reader populations',s1['consultant_reader_count']==s2['populations']['consultant']==9 and s1['all_clinician_reader_count']==s2['populations']['all_clinicians']==20),
('gaze-only training spec',s2['trainable_layer']==fm['trainable_layer']=='conv2d_10' and s2['epochs']==fm['epochs']==25 and s2['augmentations_per_training_ECG']==fm['augmentations_per_ECG']==40 and s2['gaussian_noise_sigma']==fm['gaussian_noise_sigma']==0.03 and s2['batch_size']==fm['batch_size']==16 and s2['learning_rate']==fm['learning_rate']==0.0001),
('PTB-XL 23 output classes',pa['model_class_order']==pe['classes']==rb['classes'] and pa['n_model_classes']==pe['n_classes']==23),
('PTB-XL fold counts/no patient overlap',pa['folds_1_to_9_records']==19601 and pa['fold_10_records']==pe['test_records']==pb['ECG_records']==rb['ECG_records']==2198 and pa['patient_overlap_between_folds_1_to_9_and_10']==0),
('PTB-XL patient bootstrap',pb['resampling_unit']=='patient' and rb['resampling_unit']=='patient' and pb['bootstrap_replicates']==rb['bootstrap_replicates']==3000 and pb['unique_patients']==rb['unique_patients']==1904),
('LUDB record count/resampling',la['WFDB_record_count']==lb['records']==lc['LUDB_records']==200 and lb['model_sampling_frequency_Hz']==lc['model_sampling_frequency_Hz']==100),
('LUDB endpoint lists',lb['primary_tasks']==lc['primary_tasks']==ld['primary_tasks'] and lb['exploratory_tasks']==lc['exploratory_tasks']==ld['exploratory_tasks']),
('LUDB bootstrap is distinct',ld['bootstrap_repetitions']==5000 and ld['resampling_unit'].startswith('LUDB record')),
('reviewer controls tied to frozen original models',rc['base_model_sha256']==fm['base_model_sha256'] and rc['original_consultant_model_sha256']==fm['consultant_weights_sha256'] and rc['original_all_clinician_model_sha256']==fm['all_clinician_weights_sha256']),
('historical config differentiated',historical['seed']==42 and s2['seed']==2026),
('presence of 15 author-supplied manifests',len(list((ROOT/'config').rglob('*.json')))-int((ROOT/'config/manual_geometry_source.json').exists())==15),
('joint-loss frozen input hashes',combined['base_model_sha256']==s1['base_model_sha256'] and combined['gaze_input_sha256']==s1['frozen_ecg_sha256'] and combined['gaze_target_sha256']==s1['clean_gaze_targets_sha256']),
('joint-loss development/test partition',combined['seed']==2026 and combined['n_diagnostic_train']==19601 and combined['n_diagnostic_test']==2198 and combined['diagnostic_test_fold']==10),
('joint-loss frozen training protocol',combined['joint_epochs']==4 and combined['diagnostic_batch_size']==128 and combined['gaze_batch_size']==16 and combined['gaze_augmentations_per_ecg']==40 and combined['trainable_from_layer']=='conv2d_10' and combined['learning_rate']==1e-5),
('joint-loss fixed lambda calculation',abs(combined['lambda'] - combined['initial_diagnostic_bce']/((combined['initial_consultant_gaze_mse']+combined['initial_all_clinician_gaze_mse'])/2))<1e-9),
('joint-loss permutation matches original control',all(rc['permuted_indices'][i]==s1['ecg_order'].index(combined['fixed_permutation'][e]) for i,e in enumerate(s1['ecg_order']))),
('repeated-permutation manifest scope',perm['seed']==2026 and perm['n_permutations']==len(perm['permutations'])==len(perm['mappings'])==10 and perm['generated_before_training'] is True),
('all repeated permutations distinct fixed-point-free',len({tuple(p) for p in perm['permutations']})==10 and all(sorted(p)==list(range(10)) and all(i!=j for i,j in enumerate(p)) for p in perm['permutations'])),
('repeated permutation mapping/index agreement',all(all(perm['mappings'][k][ecg]==s1['ecg_order'][perm['permutations'][k][i]] for i,ecg in enumerate(s1['ecg_order'])) for k in range(10))),
('additional permutations exclude primary control',tuple(rc['permuted_indices']) not in {tuple(p) for p in perm['permutations']}),

]
checks += [
('combined-loss original manifest fingerprint',hashlib.sha256((ROOT/'config/manifests/reviewer/combined_loss_manifest.json').read_bytes()).hexdigest()=='1f4f86ba41dae20f3d6387accc04b7d8a322919b6688a62cc4af3b103623ef27'),
('repeated-permutation original manifest fingerprint',hashlib.sha256((ROOT/'config/manifests/reviewer/permutation_manifest.json').read_bytes()).hexdigest()=='3d6a6bb721c0a9e88d426af3fefad87789fd81f34315cb5aefee2e5a108d68db'),
]
for name,ok in checks:
    print(('PASS' if ok else 'FAIL'),name)
if not all(ok for _,ok in checks):
    raise SystemExit('One or more internal checks failed')
print(f'Validated {len(checks)} manifest consistency checks. No datasets/models were executed.')
