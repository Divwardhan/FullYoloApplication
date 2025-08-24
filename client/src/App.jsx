import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Home from './components/Home';
import Auth from './components/Auth';
import Model from './components/Model';
import getValidToken from './utils/expiry';
import CreateModel from './components/CreateModel';

export default function App() {
  const token = getValidToken();

  return (
    <Router>
      <Routes>
        {/* Protected route: Home */}
        <Route
          path="/"
          element={
            token ? <Home /> : <Navigate to="/auth" replace />
          }
        />
        
        {/* Protected route: Model */}
        <Route
          path="/model/:modelId"
          element={
            token ? <Model /> : <Navigate to="/auth" replace />
          }
        />
        
        {/* Auth route: Redirect to home if already logged in */}
        <Route
          path="/auth"
          element={
            token ? <Navigate to="/" replace /> : <Auth />
          }
        />
        <Route 
          path="/admin_access"
          element={<CreateModel/>} 
        />
      </Routes>
    </Router>
  );
}
