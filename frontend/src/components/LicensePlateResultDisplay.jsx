const LicensePlateResultDisplay = ({ result, loading }) => {
  if (loading || !result) return null

  const isLicensePlateDetected = result !== "Error processing video" && result !== "No license plate detected"

  return (
    <div className="mt-6 animate-pop-in">
      <div
        className={`p-4 rounded-lg border-2 text-center ${
          isLicensePlateDetected
            ? "bg-green-900/30 border-green-400 text-green-300"
            : "bg-red-900/30 border-red-400 text-red-300"
        }`}
      >
        {isLicensePlateDetected ? (
          <div>
            <p className="text-sm text-gray-300 mb-2">License Plate Detected:</p>
            <p className="text-xl font-bold tracking-wider">{result}</p>
          </div>
        ) : (
          <p className="text-lg font-semibold">{result}</p>
        )}
      </div>
    </div>
  )
}

export default LicensePlateResultDisplay
