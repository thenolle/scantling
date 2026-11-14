export type Rule =
  | 'dependency'
  | 'spec'
  | 'lifecycle'
  | 'import'
  | 'endpoint'
  | 'model-file'
  | 'lockfile'

export interface Finding {
  readonly rule: Rule
  readonly file: string
  readonly line?: number
  readonly message: string
}
