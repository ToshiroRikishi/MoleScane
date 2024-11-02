// frontend/src/components/Sidebar.js
import React from "react";
import { Link } from "react-router-dom";

function Sidebar() {
  return (
    <div style={{ width: "200px", padding: "20px", borderRight: "1px solid #ccc" }}>
      <div style={{ marginBottom: "20px", textAlign: "center" }}>
        <div style={{
          width: "100px", height: "100px", borderRadius: "50%", border: "2px solid #000",
          display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto"
        }}>
          фото профиля
        </div>
      </div>
      <div>
        <Link to="/image-processing">Обработка изображения</Link>
      </div>
      <div>
        <Link to="/video-processing">Обработка видео</Link>
      </div>
      <div>
        <Link to="/skin-passport">Паспорт кожи</Link>
      </div>
      <div>
        <Link to="/video-real-time">Детекция в реальном времени</Link>
      </div>
    </div>
  );
}

export default Sidebar;
