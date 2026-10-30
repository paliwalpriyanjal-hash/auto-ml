// client/src/App.jsx
import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// Code-split route imports for optimal performance
const Home = lazy(() => import('./pages/Home'));
const Workspace = lazy(() => import('./pages/Workspace'));
const Overview = lazy(() => import('./pages/Overview'));
const EDA = lazy(() => import('./pages/EDA'));
const AutoML = lazy(() => import('./pages/AutoML'));
const CompareModels = lazy(() => import('./pages/CompareModels'));
const AdvancedCleaning = lazy(() => import('./pages/AdvanceCleaning'));
const SavedModels = lazy(() => import('./pages/SavedModels'));
const Predictions = lazy(() => import('./pages/Predictions'));
const Experiments = lazy(() => import('./pages/Experiments'));
const Explainability = lazy(() => import('./pages/Explainability'));
const HyperparameterTuning = lazy(() => import('./pages/HyperparameterTuning'));
const FeatureEngineering = lazy(() => import('./pages/FeatureEngineering'));
const Reports = lazy(() => import('./pages/Reports'));
const Monitoring = lazy(() => import('./pages/Monitoring'));
const Login = lazy(() => import('./pages/Login'));

// Polished fallback skeleton during route transitions
function PageLoadingFallback() {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center space-y-4 pt-20">
      <div className="w-10 h-10 border-3 border-[#8265FF] border-t-transparent rounded-full animate-spin shadow-lg shadow-[#8265FF]/20" />
      <span className="text-xs font-semibold text-[#9FA2B6] tracking-wider uppercase animate-pulse">
        Loading Studio...
      </span>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#070711] text-[#F8F9FD] flex flex-col font-sans selection:bg-[#8265FF]/30 selection:text-white">
        <Navbar />
        <main className="flex-1">
          <Suspense fallback={<PageLoadingFallback />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/overview" element={<Overview />} />
              <Route path="/eda" element={<EDA />} />
              <Route path="/automl" element={<AutoML />} />
              <Route path="/train" element={<Navigate to="/automl" replace />} />
              <Route path="/compare" element={<CompareModels />} />
              <Route path="/clean" element={<AdvancedCleaning />} />
              <Route path="/models" element={<SavedModels />} />
              <Route path="/predict" element={<Predictions />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/explain" element={<Explainability />} />
              <Route path="/tune" element={<HyperparameterTuning />} />
              <Route path="/features" element={<FeatureEngineering />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/monitoring" element={<Monitoring />} />
              <Route path="/login" element={<Login />} />
              {/* Fallback to home */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
