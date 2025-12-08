/**
 * Main Application Component
 *
 * Sets up routing and app structure
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { GamesList } from './pages/GamesList';
import { VideoAnalysis } from './pages/VideoAnalysis';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Games List Page (Home) */}
        <Route path="/" element={<GamesList />} />

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
  );
}

export default App;
