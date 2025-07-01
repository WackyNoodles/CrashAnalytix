function ResultDisplay({ result, loading, onViewDetails }) {
  if (loading || !result) {
    return null;
  }

  const NoAccident = result.includes("No Accident Detected");
  const isAccident = !NoAccident;

  return (
    <div className="mt-6 p-4 bg-cyan-900/30 rounded-lg animate-pop-in border border-cyan-500/20">
      {isAccident ? (
        <div className="flex flex-col items-center">
          <p className="text-lg font-semibold text-red-400 animate-pulse mb-2">{result}</p>
          <button
            onClick={onViewDetails}
            className="mt-2 px-4 py-2 bg-red-500/80 hover:bg-red-600/80 text-white rounded-lg transition-all transform hover:scale-[1.02] active:scale-95"
          >
            Click for more details
          </button>
        </div>
      ) : (
        <p className="text-lg font-semibold text-green-400 animate-pulse">{result}</p>
      )}
    </div>
  );
}

export default ResultDisplay;