import { useNavigate } from 'react-router-dom';

export default function Navbar() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('token');
    window.location.href = '/auth';
  };

  return (
    <nav className="bg-slate-900 border-b border-slate-700 py-4 px-6 flex items-center justify-between sticky top-0 z-50">
      <div 
        className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 cursor-pointer"
        onClick={() => navigate('/')}
      >
        YOLO Model Gallery
      </div>
      
      <div className="flex gap-4 items-center">
        <div className="hidden md:block text-slate-300 text-sm">
          Discover, compare, and test state-of-the-art models
        </div>
        
        <button
          onClick={() => navigate('/')}
          className="text-white bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-md transition-colors duration-200 border border-slate-600"
        >
          Explore Models
        </button>
        
        <button
          onClick={handleLogout}
          className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-md hover:from-blue-600 hover:to-purple-700 transition-all duration-200"
        >
            Logout
        </button>
      </div>
    </nav>
  );
}