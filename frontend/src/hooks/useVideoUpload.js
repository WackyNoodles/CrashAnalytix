import { useState, useRef } from "react";
import axios from "axios";

function useVideoUpload() {
  const [video, setVideo] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [accidentData, setAccidentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideo(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleUpload = async () => {
    if (!video) return alert("Please select a video first");

    setLoading(true);
    setResult("");
    setAccidentData(null);

    const formData = new FormData();
    formData.append("file", video);

    const startTime = performance.now();

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      const endTime = performance.now();
      const timeTaken = (endTime - startTime).toFixed(2);
      console.log(`⏱️ Prediction Time: ${timeTaken} ms`);
      console.log("Response data:", response.data);

      setLoading(false);
      setResult(response.data.result);

      // Store the complete accident data
      if (response.data.result === "Accident Detected") {
        const accidentDataWithTimestamp = {
          ...response.data,
          timestamp: response.data.timestamp || new Date().toISOString(),
          processingTime: timeTaken,
        };

        setAccidentData(accidentDataWithTimestamp);
        console.log("Accident data saved to MongoDB via backend");
      }
    } catch (error) {
      console.error("❌ Error uploading video:", error);
      setLoading(false);
      setResult("Error processing video");
    }
  };

  // Process accident data for the AccidentDetailsPage component
  const getProcessedAccidentData = () => {
    if (!accidentData) return null;

    // Process entities data
    const vehicles = accidentData.entities.filter(
      (entity) => entity.type !== "pedestrian"
    );

    const pedestrians = accidentData.entities.filter(
      (entity) => entity.type === "pedestrian"
    );

    // Extract vehicle types (capitalize first letter)
    const vehicleTypes = vehicles.map((vehicle) => {
      const type = vehicle.type;
      return type.charAt(0).toUpperCase() + type.slice(1);
    });

    // Find visible license plates
    const visibleLicensePlates = vehicles
      .map((vehicle) => vehicle.license_plate)
      .filter((plate) => plate && plate !== "undefined");

    return {
      severity: accidentData.severity, // Directly use the backend value
      vehicles: {
        count: vehicles.length,
        types: vehicleTypes,
      },
      pedestrians: pedestrians.length,
      licensePlate: visibleLicensePlates.length > 0
        ? visibleLicensePlates[0]
        : "License number not visible",
      visibleLicensePlates: visibleLicensePlates,
      timestamp: accidentData.timestamp
        ? new Date(accidentData.timestamp).toLocaleString()
        : new Date().toLocaleString(),
      accidentType: "Vehicle collision", // Still a placeholder
      classification: "Accident",
      filename: accidentData.filename || "Unknown"
    };
  };

  return {
    video,
    preview,
    result,
    loading,
    videoRef,
    handleFileChange,
    handleUpload,
    accidentData,
    getProcessedAccidentData,
  };
}

export default useVideoUpload;




//  Final Response: {'result': 'Accident Detected', 'severity': 'Minor', 'entities': [{'type': 'truck', 'license_plate': 'NWFP-893'}, {'type': 'car', 'license_plate': ''}, {'type': 'pedestrian'}], 'filepath': './collages/collage.jpg', '_id': '4dc28d38-5a8a-4973-a55d-433395ba8a0d'}