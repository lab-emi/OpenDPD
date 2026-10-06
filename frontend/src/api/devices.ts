import type { ExperimentConfigInput, Schemas } from './types'

type DeviceInfo = Schemas['DeviceInfo']

/** Expand accelerator families into selectable devices; older/hosted APIs may only supply a count. */
export function deviceOptions(devices: DeviceInfo[]) {
  return devices.flatMap(d => d.device === 'cuda' && d.detected
    ? Array.from({ length: d.count }, (_, index) => ({
      ...d,
      value: `cuda:${index}`,
      label: `GPU ${index} (cuda:${index})`,
      name: d.instances?.find(instance => instance.index === index)?.name ?? (d.count === 1 ? d.name : null),
    }))
    : [{ ...d, value: d.device, label: d.device }])
}

export function executionDevice(value: string): NonNullable<ExperimentConfigInput['execution']> {
  return value.startsWith('cuda:')
    ? { device: 'cuda', device_index: Number(value.slice(5)) }
    : { device: value as 'cpu' | 'cuda' | 'mps' }
}

export function deviceSpec(execution: ExperimentConfigInput['execution']): string {
  return execution?.device === 'cuda' ? `cuda:${execution.device_index ?? 0}` : execution?.device ?? 'cpu'
}
