import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Eye, Brain, Zap, Activity } from 'lucide-react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from './Navbar';

const MAIN_SERVER_URL = import.meta.env.VITE_MAIN_SERVER_URL;

export default function Model() {
  const { modelId } = useParams();
  const navigate = useNavigate();

  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [modelName, setModelName] = useState('Loading...');
  const [modelType, setModelType] = useState('Yolo');
  const [modelDescription, setModelDescription] = useState('A YOLO-based object detection model');

  const fileInputRef = useRef();
  const previewRef = useRef(null);
  const token = JSON.parse(localStorage.getItem('token'));

const handleAuthError = (message) => {
  console.warn('Authentication failed:', message);
  localStorage.removeItem('token');
  navigate('/auth', { 
    state: { error: message },
    replace: true 
  });
}
  useEffect(() => {
    const fetchModel = async () => {
      try {
        const res = await axios.get(`${MAIN_SERVER_URL}/api/models/${modelId}`);
        setModelName(res.data.name || `Model ${modelId}`);
        setModelDescription(res.data.description || 'A YOLO-based object detection model trained for general purpose object recognition');
        setModelType(res.data.type || 'Yolo');
        console.log(res.data);
      } catch (error) {
        if (error.response) {
          if (error.response.status === 404) {
            setModelName(`Model ${modelId} (Not Found)`);
            setModelDescription('Model not found or unavailable');
          } else if (error.response.status === 401 || error.response.status === 403) {
            handleAuthError('Invalid or expired token');
          } else {
            console.error('Error fetching model:', error.response.status);
            alert('Failed to fetch model.');
          }
        } else {
          console.error('Server unreachable');
          alert('Server unreachable');
        }
      }
    };
    fetchModel();
  }, [modelId]);

  useEffect(() => {
    if (imagePreview && previewRef.current) {
      previewRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [imagePreview]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setAnnotatedImage(null);
    setPredictions([]);
  };

  const handleSubmit = async () => {
    if (!imageFile) return;
    setLoading(true);
    setProcessingStep(0);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      console.log(token.token)
      const res = await axios.post(
        `${MAIN_SERVER_URL}/api/inference/${modelId}`,
        formData,
        {
          headers: {
            token: token.token
          }
        }
      );
      console.log('Inference response:', res.data);

      const { annotated_image_url, json_response } = res.data;
      setAnnotatedImage(annotated_image_url);

      const parsed = JSON.parse(json_response);
      const formatted = parsed.map(p => ({
        class: p.class,
        confidence: p.confidence,
        bbox: p.box
      }));

      setPredictions(formatted);
    } catch (error) {
      if (error.response) {
        if (error.response.status === 401 || error.response.status === 403) {
          handleAuthError('Token expired or invalid');
        } else if (error.response.status === 404) {
          console.error('Model not found');
          alert('Model not found.');
        } else {
          console.error('Server responded with error:', error.response.status);
          alert(`Server error: ${error.response.status}`);
        }
      } else {
        console.error('Request failed:', error.message);
        alert('Network error or server is down.');
        window.location.href = '/auth'; 
      }
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  };

  return (
  
    <div className="h-screen overflow-hidden flex flex-col bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <Navbar/>
      <header className="px-8 py-4 bg-black/30 backdrop-blur-sm border-b border-purple-500/30">
        <h1 className="text-3xl text-center font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
          {modelName}
        </h1>
        <p className="text-center text-purple-300 text-sm mt-1">{modelDescription}</p>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel - Upload */}
        <div className="w-1/4 p-4 flex flex-col border-r border-purple-500/30">
          <div
            onClick={() => fileInputRef.current.click()}
            className="flex-1 bg-gradient-to-br from-purple-900/50 to-pink-900/50 border-2 border-dashed border-purple-400/50 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-purple-400 transition-all duration-300 hover:from-purple-900/70 hover:to-pink-900/70 mb-4"
          >
            <Upload className="w-12 h-12 text-purple-400 mb-4" />
            <p className="text-lg font-semibold text-purple-200">Upload Image</p>
            <p className="text-purple-300 text-xs mt-1">Support: JPG, PNG, WebP</p>
            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              onChange={handleImageChange}
              className="hidden"
            />
          </div>

          {imagePreview && (
            <div className="flex flex-col items-center">
              <img
                src={imagePreview}
                alt="Thumbnail"
                className="w-32 h-32 object-contain border border-purple-500/30 rounded-lg mb-4"
              />
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-2 px-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Activity className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Run Detection
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Center Panel - Image Preview */}
        <div className="w-2/4 p-4 flex flex-col border-r border-purple-500/30" ref={previewRef}>
          <div className="bg-black/30 backdrop-blur-sm border border-purple-500/30 rounded-2xl p-4 flex-1 flex flex-col">
            <h2 className="text-lg font-semibold mb-2 text-purple-200">Image Preview</h2>
            <div className="flex-1 flex items-center justify-center relative overflow-hidden">
              {!imagePreview ? (
                <div className="text-center text-purple-400">
                  <Eye className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>No image uploaded</p>
                </div>
              ) : (
                <>
                  <img
                    src={annotatedImage || imagePreview}
                    alt="Preview"
                    className={`max-h-[70vh] w-auto object-contain rounded-lg transition-opacity duration-500 ${loading ? 'opacity-30' : 'opacity-100'}`}
                  />
                  {loading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 rounded-lg">
                      <Activity className="w-10 h-10 text-purple-400 animate-spin mb-2" />
                      <span className="text-purple-300 text-sm">Running inference...</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="w-1/4 p-4 flex flex-col">
          <div className="bg-black/30 backdrop-blur-sm border border-purple-500/30 rounded-2xl p-4 flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold text-purple-200">Detection Results</h2>
              {predictions.length > 0 && (
                <span className="bg-purple-600/30 text-purple-200 px-2 py-0.5 rounded-full text-xs">
                  {predictions.length} objects
                </span>
              )}
            </div>

            {predictions.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-purple-400">
                <div className="text-center">
                  <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-sm">Upload an image and click "Run Detection"</p>
                </div>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto pr-2 space-y-3">
                {predictions.map((pred, idx) => (
                  <div key={idx} className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 p-3 rounded-lg border border-purple-500/30 hover:border-purple-400/50 transition-colors duration-200">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center">
                        <p className="capitalize font-medium text-purple-100 text-sm">
                          {pred.class} <span className="text-purple-300">Detected</span>
                          <span className="w-2 h-2 bg-green-400 rounded-full inline-block ml-2"></span>
                        </p>
                      </div>
                      <span className="text-xs bg-purple-600/20 text-purple-200 px-2 py-1 rounded">
                        #{idx + 1}
                      </span>
                    </div>
                    <div className="mt-2">
                      {/* <p className="text-xs text-green-400 font-mono">
                        Confidence: {(pred.confidence * 100).toFixed(1)}%
                      </p> */}
                      <p className="text-xs text-purple-300 mt-1">
                        <span className="font-semibold">Location:</span> [{pred.bbox.map(n => n.toFixed(1)).join(', ')}]
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
    
  );
}