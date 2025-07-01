import Lottie from "lottie-react"
import Animation from "../assets/Animation.json"

function Header() {
  return (
    <>
      <div className="mt-[-100px] mb-[-70px] w-80 h-80 animate-float">
        <Lottie animationData={Animation} loop={true} />
      </div>

      <h1 className="text-5xl font-extrabold accident-title">CrashAnalytix System</h1>
      <p className="text-cyan-100 text-lg  font-medium">To begin with the analysis start off by choosing an option</p>
    </>
  )
}

export default Header
