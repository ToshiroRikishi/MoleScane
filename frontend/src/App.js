// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

// frontend/src/App.js
import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import ImageProcessing from "./components/ImageProcessing";
import VideoProcessing from "./components/VideoProcessing";
import SkinPassport from "./components/SkinPassport";
import VideoRealTime from "./components/VideoRealTime";
import HomePage from "./components/HomePage";


function App() {
  return (
    <Router>
      <div style={{ display: "flex" }}>
        <Sidebar />
        <div style={{ marginLeft: "200px", padding: "20px", flex: 1 }}>
          <Routes>
            <Route path="/" element={<HomePage />} /> {/* Главная страница */}
            <Route path="/image-processing" element={<ImageProcessing />} />
            <Route path="/video-processing" element={<VideoProcessing />} />
            <Route path="/skin-passport" element={<SkinPassport />} />
            <Route path="/video-real-time" element={<VideoRealTime />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
