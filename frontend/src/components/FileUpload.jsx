function FileUpload({ handleFileChange, preview }) {
  return (
    <label className="custom-file-input hover:shadow-cyan-500/20">
      <input type="file" onChange={handleFileChange} className="hidden" accept=".mkv,.mp4" />
      <span className="text-sm font-medium flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
            clipRule="evenodd"
          />
        </svg>
        {preview ? "Change Video" : "Select Video"}
      </span>
    </label>
  )
}

export default FileUpload
