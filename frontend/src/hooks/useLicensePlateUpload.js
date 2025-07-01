"use client"

import { useState, useRef } from "react"
import axios from "axios"

function useLicensePlateUpload() {
  const [video, setVideo] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState("")
  const [licensePlateData, setLicensePlateData] = useState(null)
  const [loading, setLoading] = useState(false)
  const videoRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setVideo(file)
      setPreview(URL.createObjectURL(file))
    }
  }

  const handleUpload = async () => {
    if (!video) return alert("Please select a video first")

    setLoading(true)
    setResult("")
    setLicensePlateData(null)

    const formData = new FormData()
    formData.append("file", video)

    const startTime = performance.now()

    try {
      // Call the new license plate detection endpoint
      const response = await axios.post("http://127.0.0.1:5000/detect-license-plate", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })

      const endTime = performance.now()
      const timeTaken = (endTime - startTime).toFixed(2)
      console.log(`⏱️ License Plate Detection Time: ${timeTaken} ms`)
      console.log("🎯 License Plate Response data:", response.data)

      setLoading(false)

      // Handle the response
      if (response.data.license_plates && response.data.license_plates.length > 0) {
        // Show the first detected license plate
        setResult(response.data.license_plates[0])
        setLicensePlateData({
          ...response.data,
          processingTime: timeTaken,
        })
        console.log(`✅ License Plate Detected: ${response.data.license_plates[0]}`)
      } else {
        setResult("No license plate detected")
        console.log("❌ No license plates found in video")
      }
    } catch (error) {
      console.error("❌ Error processing license plate:", error)
      setLoading(false)
      setResult("Error processing video")
    }
  }

  return {
    video,
    preview,
    result,
    loading,
    videoRef,
    handleFileChange,
    handleUpload,
    licensePlateData,
  }
}

export default useLicensePlateUpload
