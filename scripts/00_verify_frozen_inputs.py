"""Read-only hash audit of the exact frozen inputs reported in the manuscript.
Execute from project root; raw datasets and pretrained model are not redistributed.
"""
from pathlib import Path
import json, hashlib

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

m=json.loads(Path('config/step1_manifest.json').read_text())
for key,sha_key in [
    ('frozen_ecg_input','frozen_ecg_sha256'),
    ('clean_gaze_targets','clean_gaze_targets_sha256'),
    ('base_model','base_model_sha256'),
    ('eye_tracking_csv','eye_tracking_csv_sha256'),
]:
    p=Path(m[key]); e=m[sha_key]
    if not p.exists():
        print('MISSING',p,'expected SHA256',e)
        continue
    a=hash_file(p)
    print(('PASS' if a==e else 'FAIL'),p,'SHA256',a)
    if a!=e: raise RuntimeError('Hash mismatch: '+str(p))
print('Read-only manifest audit complete.')
