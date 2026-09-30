"""Evaluate a completely frozen matrix and publish only after the full audit.

This is a local artifact update. It does not deploy a website or perform RF
measurements. Test workers are explicitly forbidden to train missing weights.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from benchmark import audit_arena_baselines as auditing
from benchmark import run_arena_baselines as baseline
from benchmark.run_arena_distributed import freeze, units
from benchmark.report_arena_results import report as report_markdown
from opendpd.core import arena
from opendpd.services.workspace import read_json, write_json_atomic


def verification_reports(root,docs,protocol,audit):
    """Derive ancillary documentation from the same checked matrix evidence."""
    common=dict(protocol_id=protocol.protocol_id,protocol_sha256=protocol.protocol_sha256,
                training_sha256=protocol.training_sha256)
    frozen=read_json(root/'dpd-frozen.json')
    write_json_atomic(docs/'arena-data-isolation.json',dict(**common,
        refitted_checkpoints=len(frozen['checkpoints']),frozen_at=frozen['frozen_at'],
        all_weight_hashes_verified=True,test_cases=audit['completed_cases'],
        data_access=dict(training_splits=['train','val'],evaluation_split='test',
                         all_checkpoints_frozen_before_test=True),checkpoints=frozen['checkpoints']))
    isolation=docs/'arena-data-isolation.md'
    if isolation.exists():
        begin,end='<!-- arena-v6-isolation:start -->','<!-- arena-v6-isolation:end -->'
        text=isolation.read_text()
        if begin in text and end in text:
            evidence=(f"All **{len(frozen['checkpoints'])} refitted checkpoints** were frozen before "
                f"the official test phase. The full audit verifies **{audit['completed_cases']} test cases**, "
                f"the operation ledgers and **{audit['streaming_cases_replayed']} streaming replays**. "
                '[Numerical isolation evidence](arena-data-isolation.json).')
            isolation.write_text(text.split(begin,1)[0]+begin+'\n'+evidence+'\n'+end+text.split(end,1)[1])
    details=audit['streaming_replay_details']
    maxima={field:max(d[field] for d in details) for field in (
        'stored_metric_max_abs_delta_db','same_device_chunk_max_abs_metric_delta_db','dpd_chunk_max_abs_iq')}
    write_json_atomic(docs/'arena-streaming-reproduction.json',dict(**common,
        source='benchmark.audit_arena_baselines: complete_matrix',streaming_cases_replayed=len(details),
        original_chunk_samples=200,comparison_chunk_samples=137,metric_tolerance_db=.001,
        dpd_chunk_iq_tolerance=1e-4,rows=details,**maxima))
    groups=defaultdict(list)
    for d in details:groups[(d['backbone'],d['condition_id'])].append(d)
    lines=['# Arena v6 measured-input streaming reproduction','',
        f"All {len(details)} stateful cases were replayed from their frozen v6 checkpoints. "
        'The auditor compares 200- and 137-sample DPD chunks, retains the recorded PA device, '
        'checks IQ continuity and hashes, and independently recomputes complete-symbol EVM.', '',
        '| DPD | Dataset | Cases | Stored metric max difference (dB) | Chunk max difference (dB) | Chunk max IQ difference |',
        '|---|---|---:|---:|---:|---:|']
    for (key,condition),items in sorted(groups.items()):
        values=[max(d[field] for d in items) for field in maxima]
        lines.append(f'| {key} | {condition} | {len(items)} | '+ ' | '.join(f'{v:.8g}' for v in values)+' |')
    lines += ['', 'Limits are unchanged: 0.001 dB for metric replay and 1e-4 for chunk IQ. '
        'All inputs are original measured test samples. CPU and CUDA kernels are not assumed bit-equivalent.', '',
        '[Per-case evidence](arena-streaming-reproduction.json) · [Complete audit](arena-reference-audit.json)', '',
        f'Protocol SHA-256: `{protocol.protocol_sha256}`.', '',f'Training SHA-256: `{protocol.training_sha256}`.', '']
    (docs/'arena-streaming-reproduction.md').write_text('\n'.join(lines))

def complete(workspace):
    root=Path(workspace).resolve();(root/'jobs').mkdir(parents=True,exist_ok=True)
    protocol=arena.protocol()
    freeze(root,units())
    write_json_atomic(root/'protocol.json',protocol)
    def unchanged():
        if arena.protocol().protocol_sha256!=protocol.protocol_sha256:
            raise ValueError('Arena sources changed during final evaluation')
    status=root/'finalization-progress.json'
    write_json_atomic(status,dict(phase='test_evaluation',started_at=time.time(),completed_rows=0))
    observed=0
    # A single local GPU evaluation context per job, with no other test worker.
    for model in arena.bundled_backbones():
        for board in protocol.boards:
            unchanged()
            summary=baseline.evaluate(root,protocol,board,model.key,True)
            if summary and summary['status']!='succeeded':
                raise RuntimeError(f'Official test job failed: {summary}')
            observed+=1
            write_json_atomic(status,dict(phase='test_evaluation',completed_rows=observed))
            print(json.dumps(summary or dict(board=board.board_id,backbone=model.key,status='cached')),flush=True)
    unchanged()
    candidate=root/'candidate-reference.json'
    baseline.publish(root,protocol,output=candidate)
    write_json_atomic(status,dict(phase='independent_audit',completed_rows=observed))
    audit,_=auditing.audit(root,bundle_path=candidate)
    write_json_atomic(root/'final-audit.json',audit)
    if not audit['integrity_ok'] or not audit['complete'] or audit['execution_failures']:
        raise RuntimeError('Candidate bundle failed the independent audit; publication withheld')
    unchanged()
    write_json_atomic(arena.ASSETS/arena.RESULTS_FILE,read_json(candidate))
    rows=arena.load_official_rows()
    docs=Path(arena.__file__).resolve().parents[2]/'docs/performance'
    write_json_atomic(docs/'arena-reference-audit.json',audit)
    (docs/'arena-reference-results.md').write_text(report_markdown(rows,protocol,read_json(candidate)['sha256']))
    verification_reports(root,docs,protocol,audit)
    release=docs.parent/'releases/2.3.0.md'
    if release.exists():
        begin,end='<!-- arena-v6-status:start -->','<!-- arena-v6-status:end -->'
        text=release.read_text()
        if begin in text and end in text:
            first,last=text.split(begin,1)[0],text.split(end,1)[1]
            evidence=(f'The v6 matrix is complete: **{len(rows)} results, '
                f'{sum(r.completed_cases for r in rows)} test cases**. All fits were frozen before '
                f'test evaluation. The independent audit passed, including '
                f'{audit["streaming_cases_replayed"]} streaming replays and operation-count checks. '
                'See the [current results](../performance/arena-reference-results.md) and '
                '[audit](../performance/arena-reference-audit.json). Live-browser and packaging '
                'verification for this new bundle is recorded separately when completed.')
            release.write_text(first+begin+'\n'+evidence+'\n'+end+last)
    write_json_atomic(status,dict(phase='complete',completed_rows=len(rows),completed_cases=sum(r.completed_cases for r in rows),
        completed_at=time.time(),protocol_sha256=protocol.protocol_sha256,training_sha256=protocol.training_sha256,
        result_file=str(arena.ASSETS/arena.RESULTS_FILE),integrity_ok=True))
    print('Complete frozen test matrix independently audited and published locally.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True)
    complete(p.parse_args().workspace)


if __name__=='__main__':main()
