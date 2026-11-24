import * as React from 'react'
import { cn } from '@/lib/utils'

interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
  value: number
  onValueChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  label?: string
  showValue?: boolean
  formatValue?: (value: number) => string
}

function Slider({
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  showValue = true,
  formatValue,
  className,
  ...props
}: SliderProps) {
  const displayValue = formatValue ? formatValue(value) : value.toString()

  return (
    <div className={cn('space-y-2', className)}>
      {(label || showValue) && (
        <div className="flex items-center justify-between">
          {label && <span className="text-sm font-medium">{label}</span>}
          {showValue && <span className="text-sm text-muted-foreground font-mono">{displayValue}</span>}
        </div>
      )}
      <input
        type="range"
        value={value}
        onChange={(e) => onValueChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className={cn(
          'w-full h-2 rounded-full appearance-none cursor-pointer',
          'bg-secondary [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4',
          '[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-sm',
          '[&::-webkit-slider-thumb]:hover:bg-primary/90',
        )}
        {...props}
      />
    </div>
  )
}

export { Slider }
