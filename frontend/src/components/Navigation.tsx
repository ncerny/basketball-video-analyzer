/**
 * Navigation Component
 *
 * Simple navigation bar for top-level pages
 */

import { Link, useLocation } from 'react-router-dom';

export function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <nav className="bg-gray-800 border-b border-gray-700 mb-8">
      <div className="max-w-7xl mx-auto px-8">
        <div className="flex space-x-8 py-4">
          <Link
            to="/"
            className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive('/')
                ? 'bg-gray-900 text-white'
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            Games
          </Link>
          <Link
            to="/players"
            className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive('/players')
                ? 'bg-gray-900 text-white'
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            Players
          </Link>
        </div>
      </div>
    </nav>
  );
}
