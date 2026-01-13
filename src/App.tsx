import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { FaCamera, FaImages, FaCheckCircle, FaSpinner, FaLeaf, FaTimes, FaExclamationTriangle } from 'react-icons/fa';
import { GiLoincloth } from 'react-icons/gi';
import * as mobilenet from '@tensorflow-models/mobilenet';

const LABELS = ['Kawung', 'Mega_Mendung', 'Parang', 'Truntum'];

const App = () => {
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [generalModel, setGeneralModel] = useState<mobilenet.MobileNet | null>(null);
  const [prediction, setPrediction] = useState<{ label: string; confidence: string } | null>(null);

  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);

  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      setIsModelLoading(true);
      try {
        const batikM = await tf.loadGraphModel('/model/model.json');
        setModel(batikM);

        const generalM = await mobilenet.load();
        setGeneralModel(generalM);
      } catch (e) {
        console.error(e);
      } finally {
        setIsModelLoading(false);
      }
    };

    tf.ready().then(() => loadModels());
  }, []);

  const handlePredict = async () => {
    if (!imageRef.current || !model || !generalModel) return;

    setIsPredicting(true);
    setPrediction(null);

    await new Promise(resolve => setTimeout(resolve, 100));

    try {
      const generalPredictions = await generalModel.classify(imageRef.current);
      console.log("MobileNet Mendeteksi:", generalPredictions);
      const topResult = generalPredictions[0].className.toLowerCase();
      const BLOCKED_KEYWORDS = [
        // Kartun/Kertas/Layar (Musuh utama kita)
        'comic book', 'web site', 'monitor', 'screen', 'menu', 'packet',
        'envelope', 'binder', 'poster', 'scoreboard', 'slot', 'television',

        // Kategori: Manusia
        'person', 'face', 'man', 'woman', 'groom', 'suit',

        // Kategori: Kendaraan
        'car', 'truck', 'bike', 'racer', 'vehicle', 'wheel',
        // Kategori: Makanan
        'pizza', 'burger', 'hotdog', 'plate', 'ice cream'
      ];

      // Cek apakah hasil deteksi ada di daftar terlarang?
      const isBlocked = BLOCKED_KEYWORDS.some(keyword => topResult.includes(keyword));
      if (isBlocked) {
        setPrediction({
          label: `Bukan Batik (${topResult})`,
          confidence: "100%"
        });
        setIsPredicting(false);
        return;
      }

      // ---------------------------------------------------------
      // TAHAP 2: BATIK AI (ResNet)
      // ---------------------------------------------------------
      const img = tf.browser.fromPixels(imageRef.current)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims(0);

      const result = model.predict(img) as tf.Tensor;
      const data = await result.data();

      let maxScore = -1;
      let maxClass = -1;

      for (let i = 0; i < data.length; i++) {
        if (data[i] > maxScore) {
          maxScore = data[i];
          maxClass = i;
        }
      }

      // Threshold Batik (Bisa dinaikkan sedikit biar aman)
      if (maxScore > 0.98) {
        setPrediction({
          label: LABELS[maxClass],
          confidence: (maxScore * 100).toFixed(1) + '%'
        });
      } else {
        setPrediction({
          label: "Tidak Dikenali / Bukan Batik",
          confidence: (maxScore * 100).toFixed(1) + '%'
        });
      }

      tf.dispose([img, result]);

    } catch (error) {
      console.error(error);
      alert("Terjadi kesalahan.");
    } finally {
      setIsPredicting(false);
    }
  };
  //  HELPER
  const resetAll = () => {
    setImageURL(null);
    setPrediction(null);
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  };

  //  UPLOAD GAMBAR 
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
  };

  //  KAMERA 
  const handleCameraInput = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (file) processFile(file);
    };
    input.click();
  };

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      resetAll();
      setImageURL(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="bg-amber-50 min-h-screen font-sans">
      <div className="max-w-md mx-auto bg-white shadow-2xl min-h-screen flex flex-col border-x border-amber-200">

        {/* HEADER */}
        <div className="bg-gradient-to-r from-amber-800 to-amber-900 p-5 shadow-md">
          <div className="flex items-center justify-center text-amber-50">
            <GiLoincloth className="text-4xl mr-3 animate-pulse" />
            <div>
              <h1 className="text-xl font-bold tracking-wider">BATIK AI</h1>
              <p className="text-xs text-amber-200 opacity-80">Klasifikasi Motif Keraton</p>
              <p className="text-xs text-amber-200 opacity-80">BATIK (KAWUNG, MEGA MENDUNG, PARANG, TRUNTUM)</p>
            </div>
          </div>
        </div>

        {/* MAIN CONTENT */}
        <div className="p-5 flex-1 flex flex-col">

          {/* AREA GAMBAR */}
          <div className="bg-amber-100 border-2 border-dashed border-amber-400 rounded-xl p-3 mb-6 relative min-h-[300px] flex items-center justify-center overflow-hidden">
            {imageURL ? (
              <>
                <img
                  ref={imageRef}
                  src={imageURL}
                  alt="Preview Batik"
                  className="w-full h-72 object-contain rounded-lg shadow-sm bg-white"
                />
                <button
                  onClick={resetAll}
                  className="absolute top-3 right-3 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg transition-transform hover:scale-110"
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

          {/* AREA HASIL PREDIKSI */}
          {prediction && (
            <div className={`mb-6 p-4 rounded-xl shadow-sm border-l-4 animate-fade-in-up ${prediction.label.includes("Bukan")
              ? 'bg-red-50 border-red-500 text-red-800'
              : 'bg-green-50 border-green-600 text-green-900'
              }`}>
              <div className="flex items-start">
                <div className="mt-1 mr-3">
                  {prediction.label.includes("Bukan") ? <FaExclamationTriangle size={24} /> : <FaCheckCircle size={24} />}
                </div>
                <div>
                  <p className="text-xs font-bold uppercase tracking-wide opacity-70">Hasil Deteksi</p>
                  <h2 className="text-2xl font-extrabold">{prediction.label.replace('_', ' ')}</h2>
                  <p className="text-sm font-medium mt-1">
                    Tingkat Keyakinan: <span className="font-bold">{prediction.confidence}</span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TOMBOL INPUT */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <button
              onClick={handleCameraInput}
              disabled={isPredicting || isModelLoading}
              className="btn bg-amber-700 hover:bg-amber-700 transition shadow-md disabled:opacity-50 text-white"
            >
              <FaCamera size={20} />
              <span className="text-xs font-bold">Kamera</span>
            </button>

            <button
              onClick={() => document.getElementById('fileInput')?.click()}
              disabled={isPredicting || isModelLoading}
              className="btn bg-amber-700 hover:bg-amber-700 transition shadow-md disabled:opacity-50 text-white"
            >
              <FaImages size={20} />
              <span className="text-xs font-bold">Galeri</span>
            </button>

            <input
              id="fileInput"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>

          {/* TOMBOL PREDIKSI (DENGAN LOADING) */}
          <button
            onClick={handlePredict}
            disabled={!imageURL || isPredicting || isModelLoading}
            className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg flex items-center justify-center transition-all ${!imageURL || isPredicting || isModelLoading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700 text-white hover:shadow-xl transform active:scale-95'
              }`}
          >
            {isPredicting ? (
              <>
                <FaSpinner className="animate-spin mr-3 text-xl" />
                Sedang Menganalisis...
              </>
            ) : (
              <>
                <FaLeaf className="mr-2" />
                Deteksi Motif Sekarang
              </>
            )}
          </button>

          {/* STATUS MODEL FOOTER */}
          <div className="mt-auto pt-6 text-center">
            {isModelLoading ? (
              <p className="text-xs text-amber-600 flex items-center justify-center bg-amber-100 py-1 px-3 rounded-full inline-flex">
                <FaSpinner className="animate-spin mr-2" />
                Sedang memuat AI Model...
              </p>
            ) : model ? (
              <p className="text-xs text-green-600 flex items-center justify-center bg-green-100 py-1 px-3 rounded-full inline-flex">
                <FaCheckCircle className="mr-2" />
                AI Model Siap Digunakan
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