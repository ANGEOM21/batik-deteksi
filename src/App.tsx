import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { Capacitor } from '@capacitor/core';
import { FaCamera, FaImages, FaCheckCircle, FaSpinner, FaLeaf, FaTimes, FaExclamationTriangle } from 'react-icons/fa';
import { GiLoincloth } from 'react-icons/gi';
import * as mobilenet from '@tensorflow-models/mobilenet';
import toast from 'react-hot-toast';

const LABELS = ['Kawung', 'Mega_Mendung', 'Parang', 'Truntum'];

const App = () => {
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [generalModel, setGeneralModel] = useState<mobilenet.MobileNet | null>(null);
  const [prediction, setPrediction] = useState<{ label: string; confidence: string } | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isImageReady, setIsImageReady] = useState(false);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const webCameraInputRef = useRef<HTMLInputElement | null>(null);
  const webGalleryInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      setIsModelLoading(true);
      try {
        await tf.ready();

        try {
          const batikM = await tf.loadGraphModel('/model/model.json');
          setModel(batikM);
        } catch (err) {
          toast("Gagal memuat Model Batik. Pastikan file ada di folder public/model", {
            icon: <FaExclamationTriangle />,
          });
          console.error(err);
          setIsModelLoading(false);
          return;
        }

        try {
          const generalM = await mobilenet.load();
          setGeneralModel(generalM);
        } catch (err) {
          console.error("Gagal memuat MobileNet (Cek Internet)", err);
        }

      } catch (e) {
        console.error(e);
      } finally {
        setIsModelLoading(false);
      }
    };
    loadModels();
  }, []);

  const resetAll = () => {
    setImageURL(null);
    setPrediction(null);
    setIsImageReady(false);
    if (webCameraInputRef.current) webCameraInputRef.current.value = '';
    if (webGalleryInputRef.current) webGalleryInputRef.current.value = '';
  };

  const processWebFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        resetAll();
        setImageURL(event.target.result as string);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleWebInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processWebFile(file);
  };

  const handleCamera = async () => {
    if (Capacitor.isNativePlatform()) {
      try {
        const image = await Camera.getPhoto({
          quality: 90,
          allowEditing: false,
          resultType: CameraResultType.Uri,
          source: CameraSource.Camera
        });
        if (image.webPath) {
          resetAll();
          setImageURL(image.webPath);
        }
      } catch (error) {
        console.log(error);
      }
    } else {
      webCameraInputRef.current?.click();
    }
  };

  const handleGallery = async () => {
    if (Capacitor.isNativePlatform()) {
      try {
        const image = await Camera.getPhoto({
          quality: 90,
          allowEditing: false,
          resultType: CameraResultType.Uri,
          source: CameraSource.Photos
        });
        if (image.webPath) {
          resetAll();
          setImageURL(image.webPath);
        }
      } catch (error) {
        console.log(error);
      }
    } else {
      webGalleryInputRef.current?.click();
    }
  };

  const handlePredict = async () => {
    if (!imageRef.current || !model || !isImageReady) return;

    setIsPredicting(true);
    setPrediction(null);

    await new Promise(resolve => setTimeout(resolve, 50));

    try {
      if (generalModel) {
        const generalPredictions = await generalModel.classify(imageRef.current);
        const topResult = generalPredictions[0].className.toLowerCase();

        const BLOCKED_KEYWORDS = [
          'comic book', 'web site', 'monitor', 'screen', 'menu', 'packet', 'envelope',
          'binder', 'poster', 'scoreboard', 'slot', 'television', 'person', 'face',
          'man', 'woman', 'groom', 'suit', 'car', 'truck', 'bike', 'vehicle',
          'pizza', 'burger', 'hotdog', 'plate', 'food'
        ];

        const isBlocked = BLOCKED_KEYWORDS.some(keyword => topResult.includes(keyword));

        if (isBlocked) {
          setPrediction({ label: `Bukan Batik (${topResult})`, confidence: "100%" });
          setIsPredicting(false);
          return;
        }
      }

      tf.tidy(() => {
        const img = tf.browser.fromPixels(imageRef.current!)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims(0);

        const result = model.predict(img) as tf.Tensor;
        const data = result.dataSync();

        let maxScore = -1;
        let maxClass = -1;

        for (let i = 0; i < data.length; i++) {
          if (data[i] > maxScore) {
            maxScore = data[i];
            maxClass = i;
          }
        }

        const confValue = (maxScore * 100).toFixed(1) + '%';

        if (maxScore > 0.70) {
          setPrediction({ label: LABELS[maxClass], confidence: confValue });
        } else {
          setPrediction({ label: "Tidak Dikenali / Bukan Batik", confidence: confValue });
        }
      });

    } catch (error) {
      console.error(error);
      alert("Terjadi kesalahan saat deteksi.");
    } finally {
      setIsPredicting(false);
    }
  };

  const isIOS = Capacitor.getPlatform() === 'ios';

  return (
    <div className="bg-amber-50 min-h-screen font-sans">
      <div className="max-w-md mx-auto bg-white shadow-2xl min-h-screen flex flex-col border-x border-amber-200">

        <div
          className="bg-gradient-to-r from-amber-800 to-amber-900 shadow-md px-5 pb-5"
          style={{
            paddingTop: isIOS ? '5rem' : '1.25rem',
            transition: 'padding 0.3s'
          }}
        >
          <div className="flex items-center justify-center text-amber-50">
            <GiLoincloth className="text-4xl mr-3 animate-pulse" />
            <div>
              <h1 className="text-xl font-bold tracking-wider">BATIK AI</h1>
              <p className="text-xs text-amber-200 opacity-80">Klasifikasi Motif Keraton</p>
              <p className="text-xs text-amber-200 opacity-80">(KAWUNG, MEGA MENDUNG, PARANG, TRUNTUM)</p>
            </div>
          </div>
        </div>

        <div className="p-5 flex-1 flex flex-col">
          <div className="bg-amber-100 border-2 border-dashed border-amber-400 rounded-xl p-3 mb-6 relative min-h-[300px] flex items-center justify-center overflow-hidden">
            {imageURL ? (
              <>
                <img
                  ref={imageRef}
                  src={imageURL}
                  alt="Preview Batik"
                  crossOrigin='anonymous'
                  onLoad={() => setIsImageReady(true)}
                  className="w-full h-72 object-contain rounded-lg shadow-sm bg-white"
                />
                <button
                  onClick={resetAll}
                  className="absolute top-3 right-3 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg z-10"
                >
                  <FaTimes />
                </button>
              </>
            ) : (
              <div className="text-center text-amber-600 opacity-60">
                <FaImages className="text-6xl mx-auto mb-2" />
                <p className="text-sm font-medium">Pilih atau Ambil Foto Batik</p>
              </div>
            )}
          </div>

          {prediction && (
            <div className={`mb-6 p-4 rounded-xl shadow-sm border-l-4 animate-fade-in-up ${prediction.label.includes("Bukan") || prediction.label.includes("Tidak")
              ? 'bg-red-50 border-red-500 text-red-800'
              : 'bg-green-50 border-green-600 text-green-900'
              }`}>
              <div className="flex items-start">
                <div className="mt-1 mr-3">
                  {prediction.label.includes("Bukan") || prediction.label.includes("Tidak") ? <FaExclamationTriangle size={24} /> : <FaCheckCircle size={24} />}
                </div>
                <div>
                  <p className="text-xs font-bold uppercase tracking-wide opacity-70">Hasil Deteksi</p>
                  <h2 className="text-2xl font-extrabold">{prediction.label.replace('_', ' ')}</h2>
                  <p className="text-sm font-medium mt-1">
                    Confidence: <span className="font-bold">{prediction.confidence}</span>
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 mb-4">
            <button
              onClick={handleCamera}
              disabled={isPredicting || isModelLoading}
              className="btn bg-amber-700 hover:bg-amber-800 transition py-3 text-white rounded-lg flex items-center justify-center gap-2 shadow-md"
            >
              <FaCamera size={20} />
              <span className="text-xs font-bold">Kamera</span>
            </button>

            <button
              onClick={handleGallery}
              disabled={isPredicting || isModelLoading}
              className="btn bg-amber-700 hover:bg-amber-800 transition py-3 text-white rounded-lg flex items-center justify-center gap-2 shadow-md"
            >
              <FaImages size={20} />
              <span className="text-xs font-bold">Galeri</span>
            </button>

            <input
              ref={webCameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleWebInputChange}
              className="hidden"
            />

            <input
              ref={webGalleryInputRef}
              type="file"
              accept="image/*"
              onChange={handleWebInputChange}
              className="hidden"
            />
          </div>

          <button
            onClick={handlePredict}
            disabled={!imageURL || isPredicting || isModelLoading || !isImageReady}
            className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg flex items-center justify-center transition-all ${!imageURL || isPredicting || isModelLoading || !isImageReady
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700 text-white hover:shadow-xl transform active:scale-95'
              }`}
          >
            {isPredicting ? (
              <>
                <FaSpinner className="animate-spin mr-3 text-xl" />
                Menganalisis...
              </>
            ) : (
              <>
                <FaLeaf className="mr-2" />
                Deteksi Motif
              </>
            )}
          </button>

          <div className="mt-auto pt-6 text-center">
            {isModelLoading ? (
              <p className="text-xs text-amber-600 flex items-center justify-center bg-amber-100 py-1 px-3 rounded-full">
                <FaSpinner className="animate-spin mr-2" />
                Memuat Model AI...
              </p>
            ) : model ? (
              <p className="text-xs text-green-600 flex items-center justify-center bg-green-100 py-1 px-3 rounded-full">
                <FaCheckCircle className="mr-2" />
                AI Siap Digunakan
              </p>
            ) : (
              <p className="text-xs text-red-500 bg-red-100 py-1 px-3 rounded-full inline-flex">
                Gagal memuat Model
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;