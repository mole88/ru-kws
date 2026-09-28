"""Paired evaluations must score every clip from the fixed clean manifests."""
import csv
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/compare_tflite_models.py'
spec = importlib.util.spec_from_file_location('compare_tflite_models', SCRIPT)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


def test_paired_comparison_uses_clean_views_and_reports_absent_classes(tmp_path):
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    labels = {'next': 0, 'back': 1, 'unknown': 2}
    (dataset/'labels.json').write_text(json.dumps(labels))
    write_jsonl(dataset/'evaluation/synthetic_test.jsonl',
                [dict(path='a.wav',label='next',split='test'), dict(path='b.wav',label='back',split='test')])
    write_jsonl(dataset/'evaluation/vanya_test.jsonl', [dict(path='c.wav',label='next',split='test')])
    old = tmp_path/'old.tflite'; old.write_bytes(b'old')
    new = tmp_path/'new.tflite'; new.write_bytes(b'new')
    evaluator = tmp_path/'evaluate_tflite.py'; evaluator.write_text('')

    def fake_run(script, model, data_root, manifest, labels_path, output, window, threads):
        assert data_root == dataset
        rows = compare.read_jsonl(manifest)
        output.mkdir(parents=True)
        with (output/'predictions.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['record_index','path','true_label','predicted_label','correct','confidence'])
            writer.writeheader()
            for i,r in enumerate(rows):
                pred = r['label'] if model==new else 'back'
                writer.writerow(dict(record_index=i,path=r['path'],true_label=r['label'],predicted_label=pred,
                                     correct=int(pred==r['label']),confidence=.8))
        (output/'skipped.csv').write_text('record_index,path,label,reason\n',encoding='utf-8')
        report = {label: {'support': sum(r['label']==label for r in rows),
                          'f1-score': 1. if model==new and any(r['label']==label for r in rows) else 0.}
                  for label in labels}
        report['macro avg'] = {'f1-score': sum(report[label]['f1-score'] for label in labels)/len(labels)}
        compare.write_json(output/'metrics.json', {'classification_report':report,'processing_counts':{}})

    args = ['--old-model',str(old),'--new-model',str(new),'--new-dataset-root',str(dataset),
            '--output-dir',str(tmp_path/'out'),'--evaluator',str(evaluator)]
    with patch.object(compare,'run_evaluator',side_effect=fake_run):
        report=compare.main(args)
    assert report['datasets']['new_test']['metrics']['all']['improved']==1
    assert report['datasets']['vanya']['metrics']['all']['improved']==1
    assert report['datasets']['vanya']['metrics']['unknown']['count']==0
    assert report['datasets']['vanya']['supported_macro_f1']['new']==1
    assert report['datasets']['vanya']['new_macro_f1']==pytest.approx(1/3)
    assert report['datasets']['new_test']['compared_count']==2
    assert report['datasets']['new_test']['skipped_count']==0
    paired=compare.read_predictions(tmp_path/'out/new_test/paired_predictions.csv')
    assert [r['path'] for r in paired]==['a.wav','b.wav']

    changed=tmp_path/'bad.csv'
    predictions=compare.read_predictions(tmp_path/'out/new_test/new/predictions.csv')
    with changed.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows([dict(predictions[0],path='wrong.wav'), predictions[1]])
    with pytest.raises(ValueError,match='align'):
        compare.pair_predictions(tmp_path/'out/new_test/old/predictions.csv',changed,2)
    with pytest.raises(ValueError,match='counts'):
        compare.pair_predictions(tmp_path/'out/new_test/old/predictions.csv',changed,3)


def test_evaluator_uses_strict_policy_and_propagates_missing_audio(tmp_path):
    with patch.object(compare.subprocess, 'run', side_effect=FileNotFoundError('missing.wav')) as run:
        with pytest.raises(FileNotFoundError,match='missing.wav'):
            compare.run_evaluator('eval.py','model',tmp_path,'test.jsonl','labels.json',tmp_path/'out',3,2)
    command=run.call_args.args[0]
    assert command[command.index('--long-commands')+1]=='error'
    assert '--resample' not in command and '--mono' not in command
