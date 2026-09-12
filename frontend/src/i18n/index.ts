import en from './en.json'

export type MessageKey = keyof typeof en

/** All user-facing strings live in en.json (S01 decision: English default, one place to translate). */
export function t(key: MessageKey, vars: Record<string, string | number> = {}): string {
  const template: string = en[key]
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? `{${name}}`))
}
