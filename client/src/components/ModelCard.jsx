import PropTypes from 'prop-types';
import { getModelImageUrl } from '../utils/model';
import { useNavigate } from 'react-router-dom';

export default function ModelCard({ model, idx, active, setActive }) {
  const navigate = useNavigate();  // Move the hook call here

  const singleModelView = (model) => {
    navigate(`/model/${model.id}`);  // Now this works
  };

  return (
    <div
      className={`group bg-white rounded-2xl shadow-lg flex flex-col transition-transform duration-300 hover:scale-[1.03] hover:shadow-2xl cursor-pointer relative overflow-hidden border border-gray-200 ${active ? 'ring-2 ring-blue-400' : ''}`}
      style={{ minHeight: 440, maxWidth: 340 }}
      onClick={() => setActive(idx)}
      onMouseEnter={() => setActive(idx)}
      onMouseLeave={() => setActive(null)}
    >
      {/* Image as background in upper half */}
      <div
        className="w-full h-64 bg-gray-100 bg-center bg-cover transition-all duration-300 group-hover:scale-105 group-active:scale-95"
        style={{
          backgroundImage: `url('${getModelImageUrl(model)}')`,
        }}
      />
      {/* Lower half: content */}
      <div className="flex-1 flex flex-col items-center justify-between px-4 py-7">
        <h3 className="text-xl font-semibold mb-2 text-gray-800 group-hover:text-blue-700 transition-colors duration-200 text-center">
          {model.name || `Model ${idx+1}`}
        </h3>
        <p className="text-gray-600 mb-3 text-center line-clamp-2 text-base">{model.description || 'No description provided.'}</p>
        {/* <span className="text-xs text-gray-400 mb-3">Version: {model.version || 'N/A'}</span> */}
        <button onClick={() => singleModelView(model)} className="mt-2 px-5 py-2 rounded-md bg-blue-600 text-white font-medium shadow-sm transition-all duration-200 hover:bg-blue-700 active:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-300">Try Model</button>
      </div>
      <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-10 transition-opacity duration-300 bg-gradient-to-br from-gray-200 to-blue-100"></div>
    </div>
  );
}

ModelCard.propTypes = {
  model: PropTypes.object.isRequired,
  idx: PropTypes.number.isRequired,
  active: PropTypes.bool,
  setActive: PropTypes.func.isRequired,
};
