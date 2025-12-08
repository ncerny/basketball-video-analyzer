/**
 * Main Application Component
 *
 * Sets up routing and app structure
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { VideoAnalysis } from './pages/VideoAnalysis';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Video Analysis Page */}
        <Route path="/games/:gameId/analysis" element={<VideoAnalysis />} />

        {/* Default redirect - for now redirect to a test game */}
        <Route path="/" element={<Navigate to="/games/1/analysis" replace />} />

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
