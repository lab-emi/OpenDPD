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
  return <RouteError key={location.pathname}><Suspense fallback={<LoadingState />}><Outlet /></Suspense></RouteError>
}
