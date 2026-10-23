"""Measure VM residency and optional synthetic allocation/release over private SSH.

Run on the VM host as its administrator. The optional exercise touches only
anonymous synthetic memory in a separate guest process, never session data.
Use it on staging or during a drained maintenance window. No cache is flushed.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', default='opendpd-web-vm.service')
    parser.add_argument('--port', type=int, default=22224)
    parser.add_argument('--exercise-mib', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.exercise_mib and not 64 <= args.exercise_mib <= 2048:
        parser.error('--exercise-mib must be 0 or between 64 and 2048')
    details = dict(line.split('=', 1) for line in subprocess.check_output(
        ['systemctl', 'show', args.unit, '-p', 'MainPID', '-p', 'ControlGroup'], text=True).splitlines())
    pid = int(details['MainPID'])
    if pid <= 1:
        raise SystemExit('VM is not running')
    cgroup = Path('/sys/fs/cgroup') / details['ControlGroup'].lstrip('/')

    def sample():
        values = {}
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                if key in {'Rss', 'Pss', 'Anonymous'}:
                    values[key.lower() + '_mib'] = round(int(value.split()[0]) / 1024, 2)
        values['cgroup_mib'] = round(int((cgroup / 'memory.current').read_text()) / 2**20, 2)
        stats = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
        values['cgroup_file_mib'] = round(int(stats['file']) / 2**20, 2)
        values['elapsed_s'] = round(time.monotonic() - started, 2)
        return values

    code = '''import json, mmap, time
from pathlib import Path
def report(phase):
    info = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines())
    print(json.dumps(dict(phase=phase, guest_total_mib=round(info['MemTotal']/1024,2),
        guest_available_mib=round(info['MemAvailable']/1024,2))), flush=True)
report('before')
if MIB:
    block = b'\\x5a' * (2**20)
    for cycle in range(3):
        memory = mmap.mmap(-1, MIB * 2**20)
        for offset in range(MIB):
            memory[offset*2**20:(offset+1)*2**20] = block
        assert all(memory[offset*2**20] == 90 for offset in range(MIB))
        time.sleep(1)
        report('allocated_' + str(cycle + 1))
        memory.close()
        time.sleep(10)
        report('released_' + str(cycle + 1))
'''.replace('MIB', str(args.exercise_mib))
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UserKnownHostsFile=/etc/opendpd-web/known_hosts',
           '-o', 'ConnectTimeout=10', '-i', '/etc/opendpd-web/admin_ed25519',
           '-p', str(args.port), 'vmadmin@127.0.0.1', shlex.join(['python3', '-u', '-c', code])]
    started = time.monotonic()
    report = {'unit': args.unit, 'pid': pid, 'exercise_mib': args.exercise_mib,
              'scope': 'VM RSS/PSS and cgroup usage; guest capacity is not host residency.', 'samples': []}
    with subprocess.Popen(ssh, stdout=subprocess.PIPE, text=True) as process:
        try:
            for line in process.stdout:
                item = {**json.loads(line), **sample()}
                report['samples'].append(item)
                print(json.dumps(item), flush=True)
            if process.wait(timeout=15):
                raise RuntimeError('Guest memory measurement failed')
        finally:
            if process.poll() is None:
                process.kill()
    assert report['samples'], 'Guest returned no measurements'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
