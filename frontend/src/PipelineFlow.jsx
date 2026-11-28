import React, { useEffect, useRef } from 'react'

const streams = ['#ff3949', '#c91631', '#ff876c', '#ee4963', '#ffb084', '#a92740',
  '#ff5661', '#f22543', '#ff7962', '#b33754', '#ff9b81', '#ff344b']
const merge = 0.72

function position(x, index, time) {
  if (x >= merge) return 0
  const remaining = 1 - x / merge
  const envelope = Math.sin(remaining * Math.PI / 2) ** 1.5
  const phase = index * 1.94
  const wave = Math.sin(x * (8 + index % 5 * 1.7) + phase - time * .88)
  return envelope * (wave * (.54 + index % 3 * .14) + Math.cos(phase + x * 13) * .19)
}

export default function PipelineFlow() {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    let frame
    let width
    let height

    function draw(time) {
      const center = height * .5
      const amplitude = height * .45
      ctx.clearRect(0, 0, width, height)
      ctx.lineCap = 'round'

      streams.forEach((color, index) => {
        const gradient = ctx.createLinearGradient(0, 0, width * merge, 0)
        gradient.addColorStop(0, color + 'a0')
        gradient.addColorStop(.16, color + 'c0')
        gradient.addColorStop(.72, color + 'ed')
        gradient.addColorStop(1, '#ff7958')
        ctx.beginPath()
        for (let step = 0; step <= 100; step += 1) {
          const x = step / 100 * merge
          const y = center + position(x, index, time) * amplitude
          if (step === 0) ctx.moveTo(0, y)
          else ctx.lineTo(x * width, y)
        }
        ctx.strokeStyle = gradient
        ctx.lineWidth = 1.35
        ctx.stroke()
      })

      const end = .94 * width
      ctx.beginPath()
      ctx.moveTo((merge - .02) * width, center)
      ctx.lineTo(end, center)
      ctx.strokeStyle = '#ff8763'
      ctx.lineWidth = 2
      ctx.shadowColor = '#ff4c36'
      ctx.shadowBlur = 12
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(end + 9, center)
      ctx.lineTo(end, center - 5)
      ctx.lineTo(end, center + 5)
      ctx.fillStyle = '#ffac82'
      ctx.fill()

      if (!motion.matches) {
        streams.forEach((color, index) => {
          const x = (time * (.13 + index % 3 * .015) + index / streams.length) % .94
          ctx.beginPath()
          ctx.arc(x * width, center + position(x, index, time) * amplitude, 1.9, 0, 2 * Math.PI)
          ctx.fillStyle = x < merge ? color : '#ffe3cc'
          ctx.shadowColor = ctx.fillStyle
          ctx.fill()
        })
      }
      ctx.shadowBlur = 0
    }

    function animate(now) {
      draw(now / 1000)
      frame = requestAnimationFrame(animate)
    }

    function resize() {
      width = canvas.clientWidth
      height = canvas.clientHeight
      const scale = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.round(width * scale)
      canvas.height = Math.round(height * scale)
      ctx.setTransform(scale, 0, 0, scale, 0, 0)
      if (width && height) draw(0)
    }

    function syncMotion() {
      cancelAnimationFrame(frame)
      if (motion.matches) draw(0)
      else frame = requestAnimationFrame(animate)
    }

    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    resize()
    syncMotion()
    motion.addEventListener('change', syncMotion)
    return () => {
      observer.disconnect()
      cancelAnimationFrame(frame)
      motion.removeEventListener('change', syncMotion)
    }
  }, [])

  return <div className="pipeline-art" role="img" aria-label="Des flux de données convergent vers la sortie du pipeline">
    <canvas ref={ref} aria-hidden="true" />
  </div>
}
