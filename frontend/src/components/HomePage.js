// frontend/src/components/HomePage.js
import React from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css"; // Подключение CSS стилей для страницы

function HomePage() {
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate("/image-processing");
  };

  return (
    <div className="home-page">
      <div className="image-container">
        <img src="/images/faces.jpeg" alt="Faces" className="faces-image" />
      </div>
      <div className="content-container" onClick={handleButtonClick}>
        <h1>Мир дерматологии в твоих руках</h1>
        <p>Позаботься о своей коже с интеллектуальным помощником от MedVise</p>
      </div>
    </div>
  );
}

export default HomePage;
