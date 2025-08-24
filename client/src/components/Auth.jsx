import { useState } from 'react';
import { useNavigate, Navigate } from 'react-router-dom';

export default function Auth() {
  const tokenItem = localStorage.getItem('token');
  const MAIN_SERVER_URL = import.meta.env.VITE_MAIN_SERVER_URL;
  const navigate = useNavigate();

  if (tokenItem) return <Navigate to="/" replace />;

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [isSignup, setIsSignup] = useState(false);

  const saveToken = (access_token) => {
    const now = new Date();
    const tokenData = {
      token: access_token,
      expiry: now.getTime() + 2 * 24 * 60 * 60 * 1000 // 2 days
    };
    localStorage.setItem('token', JSON.stringify(tokenData));
    window.location.href = '/';

  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${MAIN_SERVER_URL}/api/user/signin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) throw new Error('Invalid login credentials');

      const data = await response.json();
      saveToken(data.access_token);
    } catch (err) {
      setError(err.message || 'Login failed');
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${MAIN_SERVER_URL}/api/user/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) throw new Error('Signup failed. Try another email.');

      const data = await response.json();
      saveToken(data.access_token);
    } catch (err) {
      setError(err.message || 'Signup failed');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-blue-50">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-full max-w-md">
        <h2 className="text-3xl font-bold text-blue-600 text-center mb-6">
          {isSignup ? 'Create an Account' : 'Login'}
        </h2>
        <form onSubmit={isSignup ? handleSignup : handleLogin} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          {error && (
            <p className="text-red-600 text-sm font-medium text-center">{error}</p>
          )}
          <button
            type="submit"
            className="w-full py-2 px-4 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition duration-300"
          >
            {isSignup ? 'Sign Up' : 'Sign In'}
          </button>
        </form>

        <div className="text-center mt-4">
          <p className="text-sm text-gray-600">
            {isSignup ? 'Already have an account?' : "Don't have an account?"}
            <button
              type="button"
              onClick={() => {
                setIsSignup(!isSignup);
                setError(null);
              }}
              className="text-blue-600 font-medium ml-1 hover:underline"
            >
              {isSignup ? 'Sign in' : 'Sign up'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
