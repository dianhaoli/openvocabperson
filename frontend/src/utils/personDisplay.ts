/** Human-readable label for a person cluster (named suspect or auto id). */
export function formatPersonDisplayLabel(
  personId: string,
  label: string | null | undefined
): string {
  return label?.trim() ? label : `Auto #${personId.slice(-6)}`;
}
