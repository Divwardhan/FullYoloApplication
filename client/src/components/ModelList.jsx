import { useEffect, useState } from 'react';
import { fetchModels } from '../utils/api';
import ModelCard from './ModelCard';
import ExtraModelCards from './ExtraModelCards';

const MAIN_SERVER_URL = import.meta.env.VITE_MAIN_SERVER_URL;

export default function ModelList() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeCard, setActiveCard] = useState(null);

  useEffect(() => {
    async function loadModels() {
      try {
        const data = await fetchModels(MAIN_SERVER_URL);
        setModels(data.models || data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadModels();
  }, []);

  return (
    <main className="container mx-auto px-4 pb-10">
      <h2 className="text-3xl font-bold mb-8 text-gray-800 text-center animate-slide-in">Available Models</h2>
      {loading && <div className="text-gray-600 text-center animate-pulse">Loading models...</div>}
      {error && <div className="text-red-500 text-center">{error}</div>}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-8">
        {models.map((model, idx) => (
          <ModelCard
            key={model.id || idx}
            model={model}
            idx={idx}
            active={activeCard === idx}
            setActive={setActiveCard}
          />
        ))}
        <ExtraModelCards name={"Windscreen Damage Detection"} image={"https://carpmai.objectstore.e2enetworks.net/yolo-inferences/9/result/d73516aa29e8457482d8ccf2d3bf5f21.jpg"} description={"Hybrid Model for detecting whether the vehicle has a damaged windscreen ."}  routelink={"https://ai.carpm.in/windscreen/"}/>
        <ExtraModelCards name={"Stoplight Functionality Detection"} image={"https://carpmai.objectstore.e2enetworks.net/yolo-inferences/9/result/8a047bb35ae54b599ee8cf1cc7b23930.png"} description={"Hybrid Model for detecting whether the vehicle has working brakelights."} routelink={"https://ai.carpm.in/stoplight/"} /> 
        
        

      </div>

    </main>
  );
} 