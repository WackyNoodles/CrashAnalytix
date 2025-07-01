import Lottie from "lottie-react"
import CarToy from "../assets/car-toy.json"

function LoadingOverlay() {
  return (
    <div className="absolute inset-0 bg-cyan-900/50 backdrop-blur-sm rounded-2xl flex items-center justify-center z-20">
      <Lottie animationData={CarToy} className="w-36 h-36 animate-bounce" loop={true} />
    </div>
  )
}

export default LoadingOverlay
