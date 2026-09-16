import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { WEB_MODE } from '@/api/client'

// The static host cannot send a frame-ancestors response header.
if (WEB_MODE && window.self !== window.top) {
  document.getElementById('root')!.textContent = 'Open OpenDPD Studio in its own browser tab.'
} else createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
