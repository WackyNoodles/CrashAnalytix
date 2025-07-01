import { useState } from "react"
import * as React from "react"
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom"
import Lottie from "lottie-react"
import Animation from "./assets/Animation.json"
import "./App.css"
import Header from "./components/Header"
import VideoPreview from "./components/VideoPreview"
import FileUpload from "./components/FileUpload"
import ResultDisplay from "./components/ResultDisplay"
import LicensePlateResultDisplay from "./components/LicensePlateResultDisplay"
import LoadingOverlay from "./components/LoadingOverlay"
import AccidentDetailsPage from "./pages/AccidentDetailsPage"
import LoadingAnimation from "./components/LoadingAnimation"
import useVideoUpload from "./hooks/useVideoUpload"
import useLicensePlateUpload from "./hooks/useLicensePlateUpload"
import AccidentHistory from "./pages/AccidentHistory"
import AccidentDetailsView from "./pages/AccidentDetailsView"

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainApp />} />
        <Route path="/detect" element={<DetectPage />} />
        <Route path="/license-plate" element={<LicensePlateDetectPage />} />
        <Route path="/accident-history" element={<AccidentHistory />} />
        <Route path="/accident-details/:id" element={<AccidentDetailsView />} />
      </Routes>
    </Router>
  )
}

// Navbar Component 
function Navbar() {
  const navigate = useNavigate()

  const handleViewHistory = () => {
    navigate("/accident-history")
  }

  const handleUploadFootage = () => {
    navigate("/detect")
  }

  const handleLicensePlateDetection = () => {
    navigate("/license-plate")
  }

  const handleHomeClick = () => {
    navigate("/")
  }

  return (
    <header className="bg-gradient-to-r from-red-900 to-red-700 p-6 shadow-lg">
      <div className="container mx-auto flex items-center justify-between">
       
        <div className="flex items-center gap-4">
          <svg
            className="w-8 h-8 text-white hover:text-red-100 transition-colors cursor-pointer"
            viewBox="0 0 24 24"
            fill="currentColor"
            onClick={handleHomeClick}
          >
            <path d="1.png" />
          </svg>

          <div
            className="text-2xl font-bold  text-white cursor-pointer hover:text-red-100 transition-colors"
            onClick={handleHomeClick}
          >
            CrashAnalytix
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleUploadFootage}
            className="px-4 py-2 bg-gradient-to-r from-[#ff7e5f] to-[#feb47b] text-white font-semibold rounded-lg 
             hover:from-[#ff6a6a] hover:to-[#ff7e5f] transition-all duration-300
             shadow-lg hover:shadow-xl hover:shadow-orange-500/30
             transform hover:scale-105 active:scale-95"
          >
            Upload footage for analysis
          </button>
          {/* <button
            onClick={handleUploadFootage}
            className="text-white"
          >
            Upload footage for analysis
          </button> */}

          <button
            onClick={handleLicensePlateDetection}
            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold rounded-lg 
             hover:from-purple-700 hover:to-indigo-700 transition-all duration-300
             shadow-lg hover:shadow-xl hover:shadow-purple-500/30
             transform hover:scale-105 active:scale-95"
          >
            License Plate Recognition
          </button>
          {/* <button
            onClick={handleLicensePlateDetection}
            className="text-white"
          >
            License Plate Recognition
          </button> */}

          <button
            onClick={handleViewHistory}
            className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-semibold rounded-lg 
             hover:from-cyan-700 hover:to-blue-700 transition-all duration-300
             shadow-lg hover:shadow-xl hover:shadow-cyan-500/30
             transform hover:scale-105 active:scale-95"
          >
            View Accident History
          </button>
          {/* <button
            onClick={handleViewHistory}
            className="text-white"
          >
            View Accident History
          </button> */}
        </div>
      </div>
    </header>
  )
}

function MainApp() {
  return (
    <div className="min-h-screen bg-cover bg-center">
      <Navbar />

      <div className="flex items-center justify-center min-h-[calc(100vh-88px)] px-8">
        <div className="bg-cyan-900/20 backdrop-blur-xs rounded-2xl p-12 shadow-2xl flex items-center gap-12 max-w-6xl">
          <div className="flex-1">
            <h1 className="text-7xl font-bold bg-gradient-to-r from-orange-400 via-red-500 to-pink-300 bg-clip-text text-transparent mb-2 font-serif leading-tight">
              CrashAnalytix
            </h1>
            
            <p className="text-white/90 text-lg leading-relaxed max-w-lg">
              An AI-powered accident detection and analysis system designed to process video input, detect accidents,
              and extract meaningful insights from detected events. The system leverages machine learning models (YOLO)
              to classify accidents, analyze severity, identify involved entities (vehicles/pedestrians), recognize
              license plates, and generate concise downloadable PDF reports.
            </p>
            <div className="mt-6 flex items-center gap-2 text-sm text-white/70">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>AI-Powered • YOLO Detection • PDF Reports</span>
            </div>
          </div>

          <div className="flex-1 flex justify-center">
            <div className="w-[450px] h-[450px]">
              <Lottie animationData={Animation} loop={true} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function DetectPage() {
  const navigate = useNavigate()
  const { video, preview, result, loading, videoRef, handleFileChange, handleUpload, getProcessedAccidentData } =
    useVideoUpload()

  const [showDetailsPage, setShowDetailsPage] = useState(false)
  const [showAnimation, setShowAnimation] = useState(false)

  React.useEffect(() => {
    document.body.classList.add("detect-page")
    return () => {
      document.body.classList.remove("detect-page")
    }
  }, [])

  const handleViewDetails = () => {
    setShowAnimation(true)
    setTimeout(() => {
      setShowAnimation(false)
      setShowDetailsPage(true)
    }, 4000)
  }

  const handleBackToHome = () => {
    setShowDetailsPage(false)
  }

  if (showAnimation) {
    return <LoadingAnimation />
  }

  if (showDetailsPage) {
    const processedAccidentData = getProcessedAccidentData()
    return <AccidentDetailsPage onBack={handleBackToHome} accidentData={processedAccidentData} />
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-cover bg-center overflow-hidden">
      <div className="relative z-10 flex flex-col items-center text-center px-4">
        <Header />

        <button
          onClick={() => navigate("/")}
          className="mb-4 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg 
           transition-all duration-300 shadow-md hover:shadow-lg
           transform hover:scale-105 active:scale-95"
        >
          ← Back to Home
        </button>

        <div className="glass-container-static bg-cyan-900/20 backdrop-blur-xl rounded-2xl p-8 shadow-2xl w-full max-w-md relative">
          {loading && <LoadingOverlay />}

          <VideoPreview preview={preview} videoRef={videoRef} />
          <FileUpload handleFileChange={handleFileChange} preview={preview} />

          <button
            onClick={handleUpload}
            disabled={loading}
            className="w-full mt-6 bg-gradient-to-br from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold py-3 px-6 rounded-xl transition-all 
                      transform hover:scale-[1.02] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
                      shadow-lg hover:shadow-cyan-500/40 relative z-10"
          >
            {loading ? "Analyzing..." : "Detect Collision"}
          </button>

          <ResultDisplay result={result} loading={loading} onViewDetails={handleViewDetails} />
        </div>
      </div>
    </div>
  )
}

function LicensePlateDetectPage() {
  const navigate = useNavigate()
  const { video, preview, result, loading, videoRef, handleFileChange, handleUpload } = useLicensePlateUpload()

  React.useEffect(() => {
    document.body.classList.add("license-plate-page")
    return () => {
      document.body.classList.remove("license-plate-page")
    }
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-cover bg-center overflow-hidden">
      <div className="relative z-10 flex flex-col items-center text-center px-4">
        <Header />

        <button
          onClick={() => navigate("/")}
          className="mb-4 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg 
           transition-all duration-300 shadow-md hover:shadow-lg
           transform hover:scale-105 active:scale-95"
        >
          ← Back to Home
        </button>

        <div className="glass-container-static bg-purple-900/20 backdrop-blur-xl rounded-2xl p-8 shadow-2xl w-full max-w-md relative">
          {loading && <LoadingOverlay />}

          <VideoPreview preview={preview} videoRef={videoRef} />
          <FileUpload handleFileChange={handleFileChange} preview={preview} />

          <button
            onClick={handleUpload}
            disabled={loading}
            className="w-full mt-6 bg-gradient-to-br from-purple-500 to-indigo-600 hover:from-purple-400 hover:to-indigo-500 text-white font-semibold py-3 px-6 rounded-xl transition-all 
                      transform hover:scale-[1.02] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
                      shadow-lg hover:shadow-purple-500/40 relative z-10"
          >
            {loading ? "Processing..." : "Extract License Plate"}
          </button>

          <LicensePlateResultDisplay result={result} loading={loading} />
        </div>
      </div>
    </div>
  )
}

export default App