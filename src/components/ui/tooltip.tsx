import * as React from 'react'
import { cn } from '@/lib/utils'

interface TooltipProps {
  content: string
  children: React.ReactNode
  side?: 'top' | 'right' | 'bottom' | 'left'
}

function Tooltip({ content, children, side = 'top' }: TooltipProps) {
  const [show, setShow] = React.useState(false)

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  }

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div
          className={cn(
            'absolute z-50 rounded-md bg-popover px-3 py-1.5 text-xs text-popover-foreground shadow-md border border-border animate-fade-in whitespace-nowrap',
            positionClasses[side],
          )}
        >
          {content}
        </div>
      )}
    </div>
  )
}

export { Tooltip }
