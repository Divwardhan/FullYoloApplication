import { useState } from 'react';

export default function CreateModel() {
  const [name, setName] = useState('');
  const [image, setImage] = useState('');
  const [type, setType] = useState('Yolo');
  const [description, setDescription] = useState(''); // ✅ New field
  const [weightFile, setWeightFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const MAIN_SERVER_URL = import.meta.env.VITE_MAIN_SERVER_URL;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const formData = new FormData();
    formData.append('name', name);
    formData.append('image', image);
    formData.append('type', type);
    formData.append('description', description); // ✅ New field
    formData.append('weight_file', weightFile);

    try {
      console.log('Submitting form data:' , formData);
      const res = await fetch(`${MAIN_SERVER_URL}/api/models/`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Failed to create model');

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    }
  };

  return (
    <div className="max-w-xl mx-auto bg-white p-8 rounded-lg shadow mt-10">
      <h2 className="text-2xl font-bold mb-6 text-blue-600">Create New Model</h2>
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block text-sm font-medium text-gray-700">Model Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-md"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Image URL</label>
          <input
            type="url"
            value={image}
            onChange={(e) => setImage(e.target.value)}
            required
            className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-md"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Model Type</label>
          <select
            value={type}
            onChange={(e) => setType(e.target.value)}
            className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-md"
          >
            <option value="Yolo">Yolo</option>
            <option value="Windscreen">Windscreen</option>
            <option value="StopLight">StopLight</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-md"
            placeholder="Enter model description (optional)"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Weight File (.pt)</label>
          <input
            type="file"
            accept=".pt"
            onChange={(e) => setWeightFile(e.target.files[0])}
            required
            className="mt-1 w-full"
          />
        </div>

        {error && <p className="text-red-600 text-sm">{error}</p>}

        <button
          type="submit"
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
        >
          Submit
        </button>
      </form>

      {response && (
        <div className="mt-6 bg-green-50 border border-green-200 rounded p-4 text-green-700">
          <h3 className="font-semibold">Model Created:</h3>
          <p><strong>ID:</strong> {response.id}</p>
          <p><strong>Name:</strong> {response.name}</p>
          <p><strong>Type:</strong> {response.type}</p>
          <p><strong>Image:</strong> <a href={response.image} className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">{response.image}</a></p>
          <p><strong>Weight File:</strong> {response.weight_file}</p>
          {response.description && (
            <p><strong>Description:</strong> {response.description}</p>
          )}
        </div>
      )}
    </div>
  );
}
