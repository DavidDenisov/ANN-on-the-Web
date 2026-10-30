"""Validate and summarize the saved model startup and length-controlled timings."""
import json
import sys
import math
import statistics
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit('Usage: python3 code/benchmarks/summarize_inference.py RUN_DIRECTORY')
run = Path(sys.argv[1]).resolve()
inputs = json.loads((run / 'inputs.json').read_text())
results = json.loads((run / 'results.json').read_text())
assert len(inputs['passages']) == len({p['id'] for p in inputs['passages']}) == 100
assert inputs['eligible_count'] == 616 and inputs['corpus_count'] == 100000
assert all(isinstance(p['id'], int) and 0 <= p['id'] < inputs['corpus_count'] for p in inputs['passages'])
assert all(len(p['text'].split()) == p['words'] >= 256 for p in inputs['passages'])
assert inputs['lengths'] == [64, 128, 256]
assert inputs['modelRevision'] == '751bff37182d3f1213fa05d7196b954e230abad9'
assert len(results['startups']) == inputs['startup_trials'] == 100
assert len(results['trials']) == len(inputs['trials']) == 1000
assert [sum(t['tokens'] == tokens for t in inputs['trials']) for tokens in inputs['lengths']] == [334, 333, 333]
scheduled = [(t['repeat'], t['input'], t['tokens']) for t in inputs['trials']]
assert all(isinstance(repeat, int) and repeat >= 0
           and isinstance(position, int) and 0 <= position < 100 and tokens in inputs['lengths']
           for repeat, position, tokens in scheduled)
assert len(set(scheduled)) == len(scheduled)
assert [(t['repeat'], t['input'], t['tokens']) for t in results['trials']] == scheduled
assert len(results['lengths']) == 300
assert {(item['id'], item['tokens']) for item in results['lengths']} == {
    (p['id'], tokens) for p in inputs['passages'] for tokens in (64, 128, 256)
}


def stats(values):
    assert values and all(math.isfinite(value) and value > 0 for value in values)
    ordered = sorted(values)
    return {'samples': len(values), 'mean_ms': statistics.mean(values),
            'p95_ms': ordered[math.ceil(len(ordered) * .95) - 1],
            'min_ms': ordered[0], 'max_ms': ordered[-1]}


summary = {'seed': inputs['seed'], 'model_revision': inputs['modelRevision'],
           'corpus_count': inputs['corpus_count'], 'eligible_count': inputs['eligible_count'],
           'sampled_passages': len(inputs['passages']), 'lengths': {}}
assert results['threads'] == 4
assert results['model'] == 'Xenova/all-MiniLM-L6-v2' and results['dtype'] == 'q8'
assert results['transformers'] == '4.3.1' and results['browserVersion'].startswith('155.')
assert 'HeadlessChrome' not in results['userAgent']
assert all(t['threads'] == results['threads'] and t['crossOriginIsolated'] for t in results['startups'])
summary.update({'threads': results['threads'], 'requested_threads': results['requestedThreads'],
                'transformers': results['transformers'],
                'cross_origin_isolated': True, 'measured_at': results['measuredAt'],
                'cpu': results['cpu'], 'browser_version': results['browserVersion']})
for tokens in inputs['lengths']:
    trials = [trial for trial in results['trials'] if trial['tokens'] == tokens]
    assert len(trials) == sum(trial['tokens'] == tokens for trial in inputs['trials'])
    assert all(trial['modelInputTokens'] == tokens for trial in trials)
    assert all(trial['id'] == inputs['passages'][trial['input']]['id'] for trial in trials)
    summary['lengths'][str(tokens)] = stats([trial['inferenceMs'] for trial in trials])
for name in ('libraryImportMs', 'downloadMs', 'initializationMs'):
    summary[name] = stats([trial[name] for trial in results['startups']])
for name in ('modelBytes', 'runtimeBytes'):
    values = {trial[name] for trial in results['startups']}
    assert len(values) == 1
    summary[name] = values.pop()
assert all(len(trial['assets']) == 6 for trial in results['startups'])
assert all(trial['modelBytes'] + trial['runtimeBytes'] == sum(a['bytes'] for a in trial['assets'])
           for trial in results['startups'])
(run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
