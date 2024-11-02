// // frontend/src/components/ImageProcessing.js
// import React from "react";

// function ImageProcessing() {
//   return <h1>Обработка видео</h1>;
// }

// export default ImageProcessing;

// VideoProcessing.js
// async function handleVideoUpload(event) {
//   const fileInput = event.target;
//   const file = fileInput.files[0];

//   if (!file) return;

//   // Отправляем видео на сервер
//   const formData = new FormData();
//   formData.append("file", file);

//   const response = await fetch("/detect_video", {
//       method: "POST",
//       body: formData,
//   });
  
//   // "http://127.0.0.1:8000/api/video/detect_video"

//   if (response.ok) {
//       // Создаем URL для видео и отображаем его на странице
//       const videoBlob = await response.blob();
//       const videoUrl = URL.createObjectURL(videoBlob);

//       const videoElement = document.getElementById("outputVideo");
//       videoElement.src = videoUrl;
//       videoElement.load();
//       videoElement.play();
//   } else {
//       console.error("Ошибка при обработке видео");
//   }
// }

// // HTML
// document.addEventListener("DOMContentLoaded", () => {
//   const videoInput = document.getElementById("videoInput");
//   videoInput.addEventListener("change", handleVideoUpload);
// });

// frontend/src/components/VideoProcessing.js

import React, { useRef } from "react";

function VideoProcessing() {
  // Используем useRef для доступа к элементам videoInput и outputVideo
  const videoInputRef = useRef(null);
  const outputVideoRef = useRef(null);

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Отправляем видео на сервер
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/video/detect_video", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        // Создаем URL для видео и отображаем его на странице
        const videoBlob = await response.blob();
        const videoUrl = URL.createObjectURL(videoBlob);
        
        // Устанавливаем URL для видео и запускаем его воспроизведение
        outputVideoRef.current.src = videoUrl;
        outputVideoRef.current.load();
        outputVideoRef.current.play();
      } else {
        console.error("Ошибка при обработке видео");
      }
    } catch (error) {
      console.error("Ошибка:", error);
    }
  };

  return (
    <div>
      <h1>Загрузите видео для детекции</h1>
      <input type="file" ref={videoInputRef} onChange={handleVideoUpload} accept="video/*" />

      <h2>Детектированное видео</h2>
      <video ref={outputVideoRef} controls width="640" height="360"></video>
    </div>
  );
}

export default VideoProcessing;
