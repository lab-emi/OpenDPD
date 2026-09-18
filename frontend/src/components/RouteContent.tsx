import { Component, Suspense, type ReactNode } from 'react'
import { Outlet, useLocation } from 'react-router'
import { ErrorState, LoadingState } from './StateBlock'

export class RouteError extends Component<{ children: ReactNode }, { error: Error | null }> {
  override state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  override render() {
    // A failed dynamic import stays rejected in React.lazy. Reloading fetches
    // the current entry manifest after a deployment or a dropped connection.
    return this.state.error ? <ErrorState error={this.state.error} onRetry={() => window.location.reload()} /> : this.props.children
  }
}

/** Keep navigation and the workflow visible while the selected page loads. */
export function RouteContent() {
  const location = useLocation()
  // Generate and Preview are views of one workspace. Keep their shared draft
  // mounted while switching views; leaving the generator still resets the boundary.
  const pageKey = location.pathname === '/signal-generator' || location.pathname.startsWith('/signal-generator/')
    ? '/signal-generator' : location.pathname
  return <RouteError key={pageKey}><Suspense fallback={<LoadingState />}><Outlet /></Suspense></RouteError>
}
