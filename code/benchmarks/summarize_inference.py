"""Validate and summarize the saved model startup and length-controlled timings."""
import json
import sys
import math
import statistics
from pathlib import Path


def stats(values) -> dict[str, float | int]:
    if not (values and all(math.isfinite(value) and value > 0 for value in values)):
        raise ValueError('Timing samples must be finite and positive')
    ordered = sorted(values)
    return {'samples': len(values), 'mean_ms': statistics.mean(values),
            'p95_ms': ordered[math.ceil(len(ordered) * .95) - 1],
            'min_ms': ordered[0], 'max_ms': ordered[-1]}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 code/benchmarks/summarize_inference.py RUN_DIRECTORY')
    run = Path(sys.argv[1]).resolve()
    inputs = json.loads((run / 'inputs.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    if not (len(inputs['passages']) == len({p['id'] for p in inputs['passages']}) == 100):
        raise ValueError('Expected 100 distinct sampled passages')
    if not (inputs['eligible_count'] == 616 and inputs['corpus_count'] == 100000):
        raise ValueError('Input corpus and eligible passage counts changed')
    if not all(isinstance(p['id'], int) and 0 <= p['id'] < inputs['corpus_count'] for p in inputs['passages']):
        raise ValueError('Passage IDs must be integers within the corpus')
    if not all(len(p['text'].split()) == p['words'] >= 256 for p in inputs['passages']):
        raise ValueError('Passage lengths do not match the recorded text')
    if inputs['lengths'] != [64, 128, 256]:
        raise ValueError('Expected token lengths 64, 128 and 256')
    if inputs['modelRevision'] != '751bff37182d3f1213fa05d7196b954e230abad9':
        raise ValueError('Unexpected model revision')
    if not (len(results['startups']) == inputs['startup_trials'] == 100):
        raise ValueError('Expected 100 startup trials')
    if not (len(results['trials']) == len(inputs['trials']) == 1000):
        raise ValueError('Expected 1,000 inference trials')
    if [sum(t['tokens'] == tokens for t in inputs['trials']) for tokens in inputs['lengths']] != [334, 333, 333]:
        raise ValueError('Token-length trial allocation changed')
    scheduled = [(t['repeat'], t['input'], t['tokens']) for t in inputs['trials']]
    if not all(isinstance(repeat, int) and repeat >= 0
               and isinstance(position, int) and 0 <= position < 100 and tokens in inputs['lengths']
               for repeat, position, tokens in scheduled):
        raise ValueError('Invalid inference trial schedule')
    if len(set(scheduled)) != len(scheduled):
        raise ValueError('Repeated inference trial in schedule')
    if [(t['repeat'], t['input'], t['tokens']) for t in results['trials']] != scheduled:
        raise ValueError('Measured trials do not match the schedule')
    if len(results['lengths']) != 300:
        raise ValueError('Expected 300 token-length measurements')
    if {(item['id'], item['tokens']) for item in results['lengths']} != {
        (p['id'], tokens) for p in inputs['passages'] for tokens in (64, 128, 256)
    }:
        raise ValueError('Token-length measurements do not cover the sampled passages')

    summary = {'seed': inputs['seed'], 'model_revision': inputs['modelRevision'],
               'corpus_count': inputs['corpus_count'], 'eligible_count': inputs['eligible_count'],
               'sampled_passages': len(inputs['passages']), 'lengths': {}}
    if results['threads'] != 4:
        raise ValueError('Expected four inference threads')
    if not (results['model'] == 'Xenova/all-MiniLM-L6-v2' and results['dtype'] == 'q8'):
        raise ValueError('Unexpected embedding model or dtype')
    if not (results['transformers'] == '4.3.1' and results['browserVersion'].startswith('155.')):
        raise ValueError('Unexpected Transformers or browser version')
    if 'HeadlessChrome' in results['userAgent']:
        raise ValueError('Expected the regular Chromium benchmark')
    if not all(t['threads'] == results['threads'] and t['crossOriginIsolated'] for t in results['startups']):
        raise ValueError('Startup trials must use isolated four-thread inference')
    summary.update({'threads': results['threads'], 'requested_threads': results['requestedThreads'],
                    'transformers': results['transformers'],
                    'cross_origin_isolated': True, 'measured_at': results['measuredAt'],
                    'cpu': results['cpu'], 'browser_version': results['browserVersion']})
    for tokens in inputs['lengths']:
        trials = [trial for trial in results['trials'] if trial['tokens'] == tokens]
        if len(trials) != sum(trial['tokens'] == tokens for trial in inputs['trials']):
            raise ValueError('Token-length trial count does not match the schedule')
        if not all(trial['modelInputTokens'] == tokens for trial in trials):
            raise ValueError('Actual model token lengths do not match the schedule')
        if not all(trial['id'] == inputs['passages'][trial['input']]['id'] for trial in trials):
            raise ValueError('Inference trial references a different passage')
        summary['lengths'][str(tokens)] = stats([trial['inferenceMs'] for trial in trials])
    for name in ('libraryImportMs', 'downloadMs', 'initializationMs'):
        summary[name] = stats([trial[name] for trial in results['startups']])
    for name in ('modelBytes', 'runtimeBytes'):
        values = {trial[name] for trial in results['startups']}
        if len(values) != 1:
            raise ValueError('Startup payload sizes changed between trials')
        summary[name] = values.pop()
    if not all(len(trial['assets']) == 6 for trial in results['startups']):
        raise ValueError('Each startup must record six asset downloads')
    if not all(trial['modelBytes'] + trial['runtimeBytes'] == sum(a['bytes'] for a in trial['assets'])
               for trial in results['startups']):
        raise ValueError('Startup byte totals do not match the asset downloads')
    temporary = run / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(run / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
