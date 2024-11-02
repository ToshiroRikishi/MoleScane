// frontend/src/components/ImageProcessing.js
// import React from "react";

// function VideoRealTime() {
//   return <h1>Детекция в реальном времени</h1>;
// }

// export default VideoRealTime;

// frontend/src/components/VideoRealTime.js
import React, { useEffect, useRef, useState } from "react";

function VideoRealTime() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isMobile, setIsMobile] = useState(false);
  const [webSocket, setWebSocket] = useState(null);
  const [detecting, setDetecting] = useState(false);

  useEffect(() => {
    // Определение типа устройства
    const userAgent = navigator.userAgent || navigator.vendor || window.opera;
    setIsMobile(/android|iPad|iPhone/i.test(userAgent.toLowerCase()));

    // Инициализация WebSocket
    const ws = new WebSocket("ws://localhost:8000/api/realTimeVideo/ws/video_real_time");
    ws.onopen = () => console.log("Connected to the WebSocket");
    ws.onclose = () => console.log("WebSocket connection closed");
    setWebSocket(ws);

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const startDetection = async () => {
    setDetecting(true);

    // Запрашиваем разрешение на использование камеры
    const stream = await navigator.mediaDevices.getUserMedia({
      video: isMobile ? { facingMode: "environment" } : true,
    });
    videoRef.current.srcObject = stream;
    videoRef.current.play();

    // Запуск отправки кадров на сервер
    videoRef.current.addEventListener("play", () => {
      const sendFrameToServer = () => {
        if (!detecting || !webSocket || webSocket.readyState !== WebSocket.OPEN) return;

        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const context = canvas.getContext("2d");
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        
        // Конвертация кадра в blob и отправка на сервер
        canvas.toBlob((blob) => {
          if (webSocket.readyState === WebSocket.OPEN && blob) {
            blob.arrayBuffer().then((buffer) => webSocket.send(buffer));
          }
        }, "image/jpeg");

        // Запрос следующего кадра
        requestAnimationFrame(sendFrameToServer);
      };
      sendFrameToServer();
    });

    // Обработка полученного кадра от сервера
    webSocket.onmessage = (event) => {
      const img = new Image();
      img.src = URL.createObjectURL(event.data);
      img.onload = () => {
        const ctx = canvasRef.current.getContext("2d");
        ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
        URL.revokeObjectURL(img.src);
      };
    };
  };

  const stopDetection = () => {
    setDetecting(false);
    const stream = videoRef.current.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (webSocket) webSocket.close();
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h2>Real-Time Mole Detection</h2>
      <video ref={videoRef} style={{ display: detecting ? "block" : "none" }} />
      <canvas ref={canvasRef} width={640} height={480} style={{ display: detecting ? "block" : "none" }} />
      {!detecting && <button onClick={startDetection}>Start Detection</button>}
      {detecting && <button onClick={stopDetection}>Stop Detection</button>}
    </div>
  );
}

export default VideoRealTime;
