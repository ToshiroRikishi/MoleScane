// // frontend/src/components/ImageProcessing.js
// import React from "react";

// function ImageProcessing() {
//   return <h1>Обработка изображения</h1>;
// }

// export default ImageProcessing;

// frontend/src/components/ImageProcessing.js
// import React, { useEffect } from "react";

// function ImageProcessing() {
//   useEffect(() => {
//     fetch("http://127.0.0.1:8000/api/image/")  // добавили завершающий слэш
//       .then(response => response.json())
//       .then(data => console.log(data))
//       .catch(error => console.error('Ошибка:', error));
//   }, []);

//   return <h1>Обработка изображения</h1>;
// }

// export default ImageProcessing;
//__________________________________________________________________________________________________

// frontend/src/components/ImageProcessing.js
// import React, { useState } from "react";
// import axios from "axios";

// function ImageProcessing() {
//     const [selectedFile, setSelectedFile] = useState(null);
//     const [processedImage, setProcessedImage] = useState(null);

//     const handleFileChange = (e) => {
//         setSelectedFile(e.target.files[0]);
//     };

//     const handleUpload = async () => {
//         if (!selectedFile) return;

//         const formData = new FormData();
//         formData.append("file", selectedFile);

//         try {
//             const response = await axios.post("http://127.0.0.1:8000/api/image/detect", formData, {
//                 headers: {
//                     "Content-Type": "multipart/form-data"
//                 },
//                 responseType: "blob"
//             });

//             // Создаем URL для изображения и отображаем его
//             const imageBlob = new Blob([response.data], { type: "image/png" });
//             const imageUrl = URL.createObjectURL(imageBlob);
//             setProcessedImage(imageUrl);
//         } catch (error) {
//             console.error("Ошибка при загрузке файла:", error);
//         }
//     };

//     return (
//         <div>
//             <h1>Обработка изображения</h1>
//             <input type="file" onChange={handleFileChange} />
//             <button onClick={handleUpload}>Обработать изображение</button>
            
//             {processedImage && (
//                 <div>
//                     <h2>Результат обработки</h2>
//                     <img src={processedImage} alt="Processed" />
//                 </div>
//             )}
//         </div>
//     );
// }

// export default ImageProcessing;


//__________________________________________________________________________________________________


// frontend/src/components/ImageProcessing.js

import React, { useState } from "react";
import axios from "axios";
import "./ImageProcessing.css"; // Подключаем CSS для стилей

function ImageProcessing() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/image/detect", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        responseType: "blob",
      });
      const imageBlob = new Blob([response.data], { type: "image/png" });
      const imageUrl = URL.createObjectURL(imageBlob);
      setProcessedImage(imageUrl);
    } catch (error) {
      console.error("Ошибка при загрузке файла:", error);
    }
  };

  return (
    <div className="processing-container">
      <h1>Обработка изображения</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Обработать изображение</button>
      {processedImage && (
        <div className="result-container">
          <h2>Результат обработки</h2>
          <img src={processedImage} alt="Processed" />
        </div>
      )}
    </div>
  );
}

export default ImageProcessing;
