/**
 * Main Application Component
 *
 * Sets up routing, Mantine provider, and app structure
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MantineProvider } from '@mantine/core';
import { GamesList } from './pages/GamesList';
import { GameDetail } from './pages/GameDetail';
import { PlayersList } from './pages/PlayersList';
import { VideoAnalysis } from './pages/VideoAnalysis';
import { theme } from './theme';

// Import Mantine styles
import '@mantine/core/styles.css';
import './App.css';

function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <BrowserRouter>
        <Routes>
          {/* Games List Page (Home) */}
          <Route path="/" element={<GamesList />} />

          {/* Game Detail Page */}
          <Route path="/games/:gameId" element={<GameDetail />} />

          {/* Players List Page */}
          <Route path="/players" element={<PlayersList />} />

          {/* Video Analysis Page */}
          <Route path="/games/:gameId/analysis" element={<VideoAnalysis />} />

          {/* 404 - Not Found */}
          <Route
            path="*"
            element={
              <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="text-center">
                  <h1 className="text-4xl font-bold mb-4">404</h1>
                  <p className="text-gray-400">Page not found</p>
                </div>
              </div>
            }
          />
        </Routes>
      </BrowserRouter>
    </MantineProvider>
  );
}

export default App;
