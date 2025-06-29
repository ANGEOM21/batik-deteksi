import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { FaCamera, FaImages, FaCheckCircle, FaSpinner, FaLeaf, FaTimes } from 'react-icons/fa';
import { GiLoincloth } from 'react-icons/gi';

const LABELS = ['Mega Mendung', 'Parang', 'Truntum', 'Kawung'];

const App = () => {
  const [imageURL, setImageURL] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [predictionLabel, setPredictionLabel] = useState<string | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  // Load model on component mount
  useEffect(() => {
    loadModel();
  }, []);

  const resetAll = () => {
    setImageURL(null);
    setPredictionLabel(null);
    // Clear file input value
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      resetAll(); // Reset sebelum menampilkan gambar baru
      setImageURL(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleCameraInput = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = () => {
          resetAll(); // Reset sebelum menampilkan gambar baru
          setImageURL(reader.result as string);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  };

  const handlePredict = async () => {
    if (!imageRef.current || !model) return;
    setIsPredicting(true);
    setPredictionLabel(null);

    try {
      const img = tf.browser.fromPixels(imageRef.current)
        .resizeNearestNeighbor([640, 640])
        .toFloat()
        .div(255)
        .expandDims(0);

      const result: any = await model.executeAsync({ x: img });
      const data = result.arraySync?.()[0];

      if (!data) return;

      const classProbs = data.slice(5);
      let maxClass = -1;
      let maxScore = -Infinity;

      for (let i = 0; i < classProbs.length; i++) {
        for (let j = 0; j < classProbs[i].length; j++) {
          const conf = classProbs[i][j];
          if (conf > maxScore) {
            maxScore = conf;
            maxClass = i;
          }
        }
      }

      const label = maxScore > 0.3 ? LABELS[maxClass] : 'Bukan Batik';
      setPredictionLabel(label);

      tf.dispose([img, result]);
    } catch (error) {
      console.error("Prediction error:", error);
      setPredictionLabel("Error saat prediksi");
    } finally {
      setIsPredicting(false);
    }
  };

  const loadModel = async () => {
    if (model) return;
    setIsModelLoading(true);
    try {
      const m = await tf.loadGraphModel('/model/model.json');
      setModel(m);
    } catch (error) {
      console.error("Model loading error:", error);
    } finally {
      setIsModelLoading(false);
    }
  };

  return (
    <div className="bg-amber-50 min-h-screen">
      <div className="bg-amber-800 shadow-lg min-h-screen flex flex-col">
        {/* Header */}
        <div className="bg-amber-900 p-4 flex items-center justify-center">
          <GiLoincloth className="text-amber-200 text-4xl mr-3" />
          <h1 className="text-2xl font-bold text-amber-100">
            Deteksi Motif Batik
          </h1>
        </div>

        {/* Main Content */}
        <div className="p-4 flex-1 flex flex-col">
          {/* Image Display */}
          <div className="bg-amber-100 border-4 border-amber-700 rounded-lg p-2 mb-4 relative">
            {imageURL ? (
              <>
                <img
                  ref={imageRef}
                  src={imageURL}
                  alt="Foto Batik"
                  className="w-full h-64 object-contain rounded-md"
                />
                <button
                  onClick={resetAll}
                  className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-2"
                  title="Reset"
                >
                  <FaTimes />
                </button>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-amber-700">
                <GiLoincloth className="text-5xl mb-3 opacity-60" />
                <p className="text-center">Belum ada gambar</p>
              </div>
            )}
          </div>

          {/* Prediction Result */}
          {predictionLabel && (
            <div className={`mb-4 p-3 rounded-lg ${predictionLabel === 'Bukan Batik'
                ? 'bg-red-100 text-red-800'
                : 'bg-green-100 text-green-800'
              }`}>
              <div className="flex items-center">
                <FaCheckCircle className="mr-2" />
                <div className='flex items-center gap-2'>
                  <p className="font-semibold">Hasil Deteksi:</p>
                  <p className="text-lg font-bold">{predictionLabel}</p>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-2 mb-4">
            <div className='w-full grid grid-cols-2 gap-10'>
              <button
                onClick={handleCameraInput}
                disabled={!!imageURL}
                className={`btn btn-lg ${imageURL ? 'btn-disabled' : 'btn-primary'} text-white`}
              >
                <FaCamera size={20} />
              </button>

              <button
                onClick={() => !imageURL && document.getElementById('fileInput')?.click()}
                disabled={!!imageURL}
                className={`btn w-full btn-lg ${imageURL ? 'btn-disabled' : 'btn-secondary'} text-white`}
              >
                <FaImages size={20} />
              </button>
            </div>

            <input
              id="fileInput"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              disabled={!!imageURL}
            />
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={isPredicting || !imageURL}
            className={`w-full py-3 rounded-lg font-bold flex items-center justify-center ${isPredicting || !imageURL
                ? 'bg-gray-300 text-gray-500'
                : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
          >
            {isPredicting ? (
              <>
                <FaSpinner className="animate-spin mr-2" />
                Memproses...
              </>
            ) : (
              <>
                <FaLeaf className="mr-2" />
                Deteksi Motif
              </>
            )}
          </button>

          {/* Model Status */}
          <div className="mt-4 text-center text-amber-100">
            {isModelLoading ? (
              <p className="flex items-center justify-center">
                <FaSpinner className="animate-spin mr-2" />
                Memuat Prediksi...
              </p>
            ) : model ? (
              <p>Prediksi siap digunakan</p>
            ) : (
              <p>Prediksi belum dimuat</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;