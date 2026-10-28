import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import Overview from './pages/Overview';
import AutoML from './pages/AutoML';
import CompareModels from './pages/CompareModels';
import AdvancedCleaning from './pages/AdvanceCleaning';
import SavedModels from './pages/SavedModels';
import Predictions from './pages/Predictions';
import Experiments from './pages/Experiments';
import Explainability from './pages/Explainability';
import HyperparameterTuning from './pages/HyperparameterTuning';
import FeatureEngineering from './pages/FeatureEngineering';
import Reports from './pages/Reports';
import Monitoring from './pages/Monitoring';
import Login from './pages/Login';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#070711] text-[#F7F7FC] flex flex-col font-sans selection:bg-[#8265FF]/30 selection:text-white">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/overview" element={<Overview />} />
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
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
