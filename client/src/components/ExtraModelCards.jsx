import React from 'react';

const ExtraModelCards = ({name , image , description , routelink}) => {
    const routeTo = () => {
        // Implement routing logic here if needed
        // For example, using useNavigate from react-router-dom
        // navigate(routelink);
        console.log("Route to: ", routelink);
        window.location.href = routelink;
    };  
  return (
    <div className="group bg-white rounded-2xl shadow-lg flex flex-col transition-transform duration-300 hover:scale-[1.03] hover:shadow-2xl cursor-pointer relative overflow-hidden border border-gray-200"
         style={{ minHeight: 440, maxWidth: 340 }}>

      {/* Image background */}
      <div
        className="w-full h-64 bg-gray-100 bg-center bg-cover transition-all duration-300 group-hover:scale-105 group-active:scale-95"
        style={{
          backgroundImage: `url(${image})`, // Replace with your static image URL
        }}
      />

      {/* Content section */}
      <div className="flex-1 flex flex-col items-center justify-between px-4 py-7">
        <h3 className="text-xl font-semibold mb-2 text-gray-800 group-hover:text-blue-700 transition-colors duration-200 text-center">
          {name}
        </h3>
        <p className="text-gray-600 mb-3 text-center line-clamp-2 text-base">
          {description}
        </p>
        {/* Optional version display */}
        {/* <span className="text-xs text-gray-400 mb-3">Version: 1.0</span> */}
        <button onClick={routeTo} className="mt-2 px-5 py-2 rounded-md bg-blue-600 text-white font-medium shadow-sm transition-all duration-200 hover:bg-blue-700 active:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-300">
          Try Model
        </button>
      </div>

      {/* Gradient overlay */}
      <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-10 transition-opacity duration-300 bg-gradient-to-br from-gray-200 to-blue-100"></div>
    </div>
  );
};

export default ExtraModelCards;
