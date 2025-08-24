import Navbar from './Navbar';
import ModelList from './ModelList';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-100">
      <Navbar />
      <header className="bg-gradient-to-r from-blue-700 to-gray-600 text-white py-10 shadow-lg mb-8 animate-fade-in">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-5xl font-extrabold mb-3 drop-shadow-lg">YOLO Model Gallery</h1>
          <p className="text-lg font-medium opacity-90">
            Discover, compare, and test state-of-the-art YOLO object detection models. Sign in to run detections!
          </p>
        </div>
      </header>
      <ModelList />
    </div>
  );
}
