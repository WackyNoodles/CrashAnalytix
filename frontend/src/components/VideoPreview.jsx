function VideoPreview({ preview, videoRef }) {
  if (!preview) {
    return (
      <div className="mb-6 aspect-video flex items-center justify-center border-2 border-dashed border-cyan-500/20 rounded-xl animate-pulse">
        <span className="text-cyan-300/50 text-sm">Preview Area</span>
      </div>
    )
  }

  return (
    <div className="relative mb-6 overflow-hidden rounded-xl aspect-video border-2 border-cyan-500/20">
      <video ref={videoRef} src={preview} controls className="w-full h-full object-cover" />
    </div>
  )
}

export default VideoPreview
