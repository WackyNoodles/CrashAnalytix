import animation from '../assets/animation.gif';
function LoadingAnimation() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-cyan-900/90 z-50">
      <div className="text-center">
        <img
          src={animation}
          alt="Loading Animation"
          className="w-96 h-64 mx-auto"
        />
        <p className="text-white text-xl mt-4 font-semibold">Fetching All Details...</p>
        <div className="mt-4 flex justify-center">
          <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div className="loading-bar h-full bg-gradient-to-r from-cyan-400 to-blue-500 animate-pulse"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default LoadingAnimation;