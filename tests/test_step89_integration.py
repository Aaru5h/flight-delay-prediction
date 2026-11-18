"""Train, reload, freeze, and evaluate all families in separate OS processes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.evaluation.artifacts_v2 import SETTINGS, read_json
from src.evaluation.calibrate_step8 import run_step8
from src.evaluation.evaluate_step9 import run_step9, verify_freeze

ROOT = Path(__file__).resolve().parents[1]


def child(code):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    root = tmp_path_factory.mktemp("step89")
    child(f"""
from pathlib import Path
import json
import pandas as pd
from tests.neural_helpers import write_step4_fixture, quick_settings
root=Path({str(root)!r})
dataset,protocol,manifest=write_step4_fixture(root,sequences=True)
for part in manifest['partitions']:
 path=dataset/part['features']
 frame=pd.read_csv(path)
 retro=frame.sched_date.ge('2013-11-01')
 frame.loc[retro,'substantial_delay']=frame.loc[retro,'flight_row_id'].str[-1].astype(int)%2
 frame.to_csv(path,index=False)
frame=pd.read_csv(dataset/'features_60m.csv')
frame.drop(columns='lead_time_minutes').to_csv(root/'legacy.csv',index=False)
legacy={{'manifest_version':'legacy-fixture','allowlist':[n for n in manifest['allowlist'] if n!='lead_time_minutes'],
        'features':[f for f in manifest['features'] if f['name']!='lead_time_minutes']}}
(root/'legacy_manifest.json').write_text(json.dumps(legacy))
quick_settings(root)
settings=json.loads(Path('configs/step5_classical.json').read_text())
settings['logistic_regression'].update(max_iter=60,iterations_per_check=10)
settings['xgboost'].update(max_estimators=12,early_stopping_rounds=3,n_jobs=1)
(root/'quick_classical.json').write_text(json.dumps(settings))
""")
    common = f"dataset_dir=Path({str(root/'step4')!r}), legacy_features=Path({str(root/'legacy.csv')!r}), legacy_manifest=Path({str(root/'legacy_manifest.json')!r}), trial_limit=1"
    child(f"from pathlib import Path; from src.models.classical.train_step5 import run_step5; run_step5({common}, output_dir=Path({str(root/'step5')!r}), settings_path=Path({str(root/'quick_classical.json')!r}))")
    child(f"from pathlib import Path; from src.models.neural.train_step6 import run_step6; run_step6({common}, output_dir=Path({str(root/'step6')!r}), settings_path=Path({str(root/'quick_neural.json')!r}))")
    child(f"from pathlib import Path; from src.models.neural.train_step7 import run_step7; run_step7(dataset_dir=Path({str(root/'step4')!r}),step6_dir=Path({str(root/'step6')!r}),output_dir=Path({str(root/'step7')!r}),settings_path=Path({str(root/'quick_neural.json')!r}),trial_limit=1,leakage_tests_passed=True)")
    settings = read_json(SETTINGS)
    settings.update(minimum_calibration_rows=4, bootstrap_resamples=25, permutation_repeats=1)
    (root / "evaluation_settings.json").write_text(json.dumps(settings))
    return root


def test_full_step89_flow_and_immutable_freeze(handoff):
    root = handoff
    with pytest.raises(ValueError, match="40-configuration"):
        run_step8(root / "step4", root / "step5", root / "step6", root / "step7",
                  root / "rejected", root / "evaluation_settings.json")
    frozen = run_step8(root / "step4", root / "step5", root / "step6", root / "step7",
                       root / "step8", root / "evaluation_settings.json", allow_reduced_budget=True)
    assert len(frozen["models"]) == 8 and frozen["development_run"]
    for item in frozen["models"].values():
        for decision in item["calibration"].values():
            assert decision["fit_month"] == "2013-10"
            assert pd.Timestamp(decision["latest_label_available_utc"]) <= pd.Timestamp(frozen["fit_cutoff_utc"])
        val = pd.read_csv(root / "step8" / item["validation_file"])
        selected = val.loc[val.calibration_source.ne("identity_no_previous_month")]
        assert (selected.calibration_source < selected.fold_id).all()
    result = run_step9(root / "step8", root / "step9", root / "reports", legacy_features=root / "legacy.csv")
    assert result["development_run"] and result["diagnostics"]["status"] == "completed"
    assert result["protocol_version"] == "2.0.0"
    assert result["raw_data_fingerprints"] == frozen["raw_data_fingerprints"]
    assert len(result["retrospective"]) == 8
    assert set(result["intervals"]) == {"60", "120", "360", "1440"}
    assert (root / "reports/lead_time_curve.png").is_file()
    assert (root / "reports/subgroups.csv").is_file()
    for key, leads in result["retrospective"].items():
        for lead, item in leads.items():
            assert item["rows"] == 24 and item["alert_budget"]["alerts"] == 2
            assert result["intervals"][lead]["metrics"][key]["average_precision"]["valid_resamples"] == 25
    assert all(not item["superiority_claim_allowed"] for item in result["success_gates"].values())
    sequence_path = root / "step4/sequences_60m.jsonl.gz"
    assert str(sequence_path) in frozen["files"]
    original_sequences = sequence_path.read_bytes()
    sequence_path.write_bytes(original_sequences + b"changed")
    with pytest.raises(ValueError, match="Frozen artifact"):
        verify_freeze(root / "step8")
    sequence_path.write_bytes(original_sequences)
    original = next(Path(p) for p in frozen["files"] if p.endswith("multi_lead_embedding_mlp.pt"))
    with original.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="Frozen artifact"):
        verify_freeze(root / "step8")
