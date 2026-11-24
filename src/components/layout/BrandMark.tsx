/** A small house made from the same connected layers as a model graph. */
export function BrandMark({ className = '' }: { className?: string }) {
  return <svg viewBox="0 0 32 32" fill="none" aria-hidden="true" className={className}>
    <path d="m6 14 10-8 10 8v12H6V14Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    <path d="m11 17 5-3 5 3-5 3-5-3Zm0 5 5 3 5-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
}
