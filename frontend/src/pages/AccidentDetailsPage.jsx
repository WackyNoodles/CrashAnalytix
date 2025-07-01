"use client"

import bgimage from "../assets/bg2.jpg"
import { useState, useEffect } from "react"
import jsPDF from "jspdf"
import accidentImage from "../assets/dummy.png"

function AccidentDetailsPage({ onBack, accidentData }) {
  const [accidentDetails, setAccidentDetails] = useState(
    accidentData || {
      severity: "moderate",
      vehicles: { count: 2, types: ["Sedan", "SUV"] },
      pedestrians: 0,
      licensePlate: "ABC-1234",
      visibleLicensePlates: ["ABC-1234"],
      timestamp: new Date().toLocaleString(),
      accidentType: "Vehicle collision",
      classification: "Accident",
    },
  )
  const [accidentImageUrl, setAccidentImageUrl] = useState(null)
  const [isImageLoading, setIsImageLoading] = useState(true)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const fetchAccidentImage = async (accidentId) => {
    try {
      setIsImageLoading(true)
      // Assuming your backend serves images with the accident ID as filename
      // Adjust the URL format based on your backend implementation
      const imageUrl = `http://localhost:5000/collages/${accidentId}.jpg`

      // Check if image exists by making a HEAD request
      const response = await fetch(imageUrl, { method: "HEAD" })
      if (response.ok) {
        setAccidentImageUrl(imageUrl)
      } else {
        console.log("No accident image found, using placeholder")
        setAccidentImageUrl(null)
      }
    } catch (error) {
      console.error("Error fetching accident image:", error)
      setAccidentImageUrl(null)
    } finally {
      setIsImageLoading(false)
    }
  }

  useEffect(() => {
    console.log("AccidentDetailsPage received data:", accidentData)
    if (accidentData) {
      console.log("Using provided accident data:", accidentData)
      setAccidentDetails(accidentData)

      // Fetch accident image if we have an ID
      if (accidentData.filename) {
        fetchAccidentImage(accidentData.filename)
      }
    }
  }, [accidentData])

  // PDF generation state
  const [isGenerating, setIsGenerating] = useState(false)

  // Get severity color based on accident severity
  const getSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case "minor":
        return "green"
      case "moderate":
        return "yellow"
      case "severe":
        return "red"
      default:
        return "yellow"
    }
  }

  // Generate PDF functionality
const generatePDF = () => {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  let y = 15;

  // ===== HEADER =====
  doc.setTextColor(0, 51, 102);
  doc.setFontSize(20);
  doc.setFont("helvetica", "bold");
  doc.text("CrashAnalytix Accident Report", pageWidth / 2, y, {
    align: "center",
  });
  y += 12;

  doc.setDrawColor(180, 180, 180);
  doc.line(margin, y, pageWidth - margin, y);
  y += 8;

  // ===== REPORT SUMMARY =====
  const summaryBoxHeight = 22;
  doc.setDrawColor(210);
  doc.setFillColor(245, 245, 245);
  doc.roundedRect(margin, y, pageWidth - 2 * margin, summaryBoxHeight, 4, 4, "FD");

  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(0, 51, 102);
  doc.text("REPORT SUMMARY", margin + 5, y + 7);

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(70, 70, 70);

  // Ensure accidentDetails.timestamp is a string
  const timestamp = accidentDetails.timestamp ? accidentDetails.timestamp : "Unknown";
  doc.text("Timestamp: " + timestamp, margin + 5, y + 13);

  y += summaryBoxHeight + 8;

  // ===== SEVERITY ASSESSMENT =====
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(0, 51, 102);
  doc.text("SEVERITY ASSESSMENT", margin, y);
  y += 8;

  const severityColors = {
    minor: [0, 153, 51],
    moderate: [255, 153, 0],
    severe: [204, 0, 0],
  };

  const textColor = severityColors[accidentDetails.severity?.toLowerCase()] || severityColors.moderate;

  doc.setFontSize(28);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...textColor);

  const displaySeverity = accidentDetails.severity ? accidentDetails.severity.charAt(0).toUpperCase() + accidentDetails.severity.slice(1) : "Unknown";

  doc.text(displaySeverity, pageWidth / 2, y, { align: "center" });
  y += 12;

  const riskBarWidth = pageWidth - 2 * margin;
  const riskBarHeight = 8;

  doc.setFillColor(0, 153, 51);
  doc.rect(margin, y, riskBarWidth * 0.3, riskBarHeight, "F");
  doc.setFillColor(255, 204, 0);
  doc.rect(margin + riskBarWidth * 0.3, y, riskBarWidth * 0.3, riskBarHeight, "F");
  doc.setFillColor(204, 0, 0);
  doc.rect(margin + riskBarWidth * 0.6, y, riskBarWidth * 0.4, riskBarHeight, "F");

  let markerPosition;
  switch (accidentDetails.severity?.toLowerCase()) {
    case "minor":
      markerPosition = 0.15;
      break;
    case "moderate":
      markerPosition = 0.45;
      break;
    case "severe":
      markerPosition = 0.8;
      break;
    default:
      markerPosition = 0.45;
  }

  doc.setFillColor(51, 51, 51);
  doc.rect(margin + riskBarWidth * markerPosition - 1, y - 2, 2, riskBarHeight + 4, "F");

  y += riskBarHeight + 5;

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(70, 70, 70);
  doc.text("Low Risk", margin, y);
  doc.text("High Risk", pageWidth - margin - 25, y, { align: "right" });
  y += 10;

  doc.setDrawColor(180, 180, 180);
  doc.line(margin, y, pageWidth - margin, y);
  y += 8;

  // ===== INCIDENT DETAILS BOX =====
  const tableBoxHeight = 45;
  doc.setDrawColor(210);
  doc.setFillColor(245, 245, 245);
  doc.roundedRect(margin, y, pageWidth - 2 * margin, tableBoxHeight, 4, 4, "FD");

  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(0, 51, 102);
  doc.text("INCIDENT DETAILS", margin + 5, y + 7);

  y += 10;

  doc.setFillColor(235, 235, 235);
  doc.rect(margin, y, pageWidth - 2 * margin, 10, "F");
  doc.setFontSize(10);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(0, 51, 102);
  doc.text("Category", margin + 5, y + 7);
  doc.text("Details", margin + 70, y + 7);
  y += 8;

  const licensePlates =
    accidentDetails.visibleLicensePlates && accidentDetails.visibleLicensePlates.length > 0
      ? accidentDetails.visibleLicensePlates.join(", ")
      : "Not visible";

  const incidentDetails = [
    {
      label: "Entities Involved",
      value: `${accidentDetails.vehicles.count} Vehicle${accidentDetails.vehicles.count !== 1 ? "s" : ""}, ${accidentDetails.pedestrians} Pedestrian${accidentDetails.pedestrians !== 1 ? "s" : ""}`,
    },
    { label: "Accident Type", value: accidentDetails.accidentType || "Vehicle Collision" },
    { label: "License Plates", value: licensePlates },
  ];

  incidentDetails.forEach((item, index) => {
    if (index % 2 === 0) {
      doc.setFillColor(250, 250, 250);
      doc.rect(margin, y, pageWidth - 2 * margin, 10, "F");
    }

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(...(index === 2 ? [204, 0, 0] : [70, 70, 70]));
    doc.text(item.label, margin + 5, y + 7);
    doc.setTextColor(...(index === 2 ? [204, 0, 0] : [40, 40, 40]));
    doc.text(item.value, margin + 70, y + 7);
    y += 8;
  });

  y += 10;

  // ===== ACCIDENT IMAGE =====
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(0, 51, 102);
  doc.text("ACCIDENT SNAPSHOT", margin, y);
  y += 15;

  const imgWidth = (pageWidth - 2 * margin) * 0.8;
  const imgHeight = (imgWidth * 9) / 16;
  const imgX = (pageWidth - imgWidth) / 2;

  const boxPadding = 4;
  doc.setDrawColor(180, 180, 180);
  doc.setLineWidth(0.8);
  doc.rect(imgX - boxPadding, y - boxPadding, imgWidth + 2 * boxPadding, imgHeight + 2 * boxPadding);

  // Use actual accident image if available, otherwise use placeholder
  const imageToUse = accidentImageUrl || accidentImage;
  doc.addImage(imageToUse, "JPG", imgX, y, imgWidth, imgHeight);
  y += imgHeight + boxPadding * 2 + 5;

  // ===== FOOTER =====
  doc.setDrawColor(180, 180, 180);
  doc.setLineWidth(0.5);
  doc.line(20, y, 190, y);

  doc.setFontSize(10);
  doc.setTextColor(100, 100, 100);
  doc.text("Generated by | CrashAnalytiX", 105, y + 10, {
    align: "center",
  });

  return doc;
};


  const handleDownload = () => {
    setIsGenerating(true)
    setTimeout(() => {
      try {
        const doc = generatePDF()
        doc.save(`crashanalytix_report_${new Date().getTime()}.pdf`)
      } catch (error) {
        console.error("PDF generation error:", error)
      } finally {
        setIsGenerating(false)
      }
    }, 300)
  }

  // Render license plates
  const renderLicensePlates = () => {
    if (accidentDetails.visibleLicensePlates && accidentDetails.visibleLicensePlates.length > 0) {
      return accidentDetails.visibleLicensePlates.map((plate, index) => (
        <div key={index} className="border-4 border-blue-800 bg-white px-6 py-3 rounded">
          <span className="text-2xl font-bold tracking-wider text-blue-900">{plate}</span>
        </div>
      ))
    } else {
      return (
        <div className="border-4 border-blue-800 bg-white px-6 py-3 rounded opacity-75">
          <span className="text-2xl font-bold tracking-wider text-blue-900">License number not visible</span>
        </div>
      )
    }
  }

  const openModal = () => {
    setIsModalOpen(true)
  }

  const closeModal = () => {
    setIsModalOpen(false)
  }

  return (
    <div
      className="min-h-screen flex flex-col bg-cover bg-center"
      style={{ backgroundImage: "url('./assets/bg4.jpg')" }}
    >
      {/* Header */}
      <header className="bg-gradient-to-r from-red-900/80 to-red-700/80 p-4 shadow-lg">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Post Analysis Details</h1>
          <button
            onClick={onBack}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-all"
          >
            Back to Dashboard
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto p-6 flex-grow">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left column */}
          <div className="space-y-6">
            <div className="bg-white/90 rounded-xl shadow-lg p-6">
              {/* Classification Output */}
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Classification Output</h2>
              <div className="flex justify-center">
                <div className="bg-red-50 p-4 rounded-lg text-center w-60">
                  <h3 className="font-bold text-red-800 mb-2">Result</h3>
                  <div className="text-xl font-bold text-red-600">Accident</div>
                </div>
              </div>
            </div>

            {/* Involved Entities */}
            <div className="bg-white/90 rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Involved Entities</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-bold text-blue-800 mb-2">Vehicles</h3>
                  <p className="text-gray-700">
                    Count: {accidentDetails.vehicles && typeof accidentDetails.vehicles.count === "number"
                      ? accidentDetails.vehicles.count
                      : 0}
                  </p>
                  <p className="text-gray-700">
                    {accidentDetails.vehicles && accidentDetails.vehicles.count > 0
                      ? `Types: ${Array.isArray(accidentDetails.vehicles.types) ? accidentDetails.vehicles.types.join(", ") : "Unknown"}`
                      : "No vehicles detected"}
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h3 className="font-bold text-purple-800 mb-2">Pedestrians</h3>
                  <p className="text-gray-700">Count: {accidentDetails.pedestrians}</p>
                  <p className="text-gray-700">
                    Status: {accidentDetails.pedestrians > 0 ? "Involved" : "None involved"}
                  </p>
                </div>
              </div>
            </div>

            {/* License Plate */}
            <div className="bg-white/90 rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">License Plate Recognition</h2>
              <div className="flex items-center justify-center bg-gray-100 p-4 rounded-lg">
                <div className="flex justify-evenly w-full max-w-4xl">License Plate not visible</div>
              </div>
              <p className="mt-4 text-gray-600 text-center">
                License plate extracted using OCR technology from the accident footage.
              </p>
            </div>
          </div>

          {/* Right column */}
          <div className="space-y-6">
            {/* Severity Score */}
            <div className="bg-white/90 rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Severity Score</h2>

              {/* Labels above scale */}
              <div className="flex justify-between text-sm font-semibold mb-1 px-1">
                <span className="text-green-600">Minor</span>
                <span className="text-yellow-600">Moderate</span>
                <span className="text-red-600">Severe</span>
              </div>

              {/* Severity Bar */}
              <div className="h-3 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 relative">
                <div
                  className={`absolute top-1/2 transform -translate-y-1/2 h-5 w-5 rounded-full border-2 border-white shadow-md ${
                    accidentDetails.severity.toLowerCase() === "minor" ||
                    accidentDetails.severity.toLowerCase() === "low"
                      ? "bg-green-600 left-[16%]"
                      : accidentDetails.severity.toLowerCase() === "moderate" ||
                          accidentDetails.severity.toLowerCase() === "medium"
                        ? "bg-yellow-600 left-[50%]"
                        : "bg-red-600 left-[84%]"
                  }`}
                ></div>
              </div>

              <p className="text-gray-600 mt-3">
                This accident has been classified as{" "}
                <span
                  className={`font-bold ${
                    accidentDetails.severity.toLowerCase() === "minor" ||
                    accidentDetails.severity.toLowerCase() === "low"
                      ? "text-green-600"
                      : accidentDetails.severity.toLowerCase() === "moderate" ||
                          accidentDetails.severity.toLowerCase() === "medium"
                        ? "text-yellow-600"
                        : "text-red-600"
                  }`}
                >
                  {accidentDetails.severity}
                </span>{" "}
                based on impact analysis.
              </p>
            </div>

            {/* Accident Snapshot */}
            <div className="bg-white/90 rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Accident Snapshot</h2>
              <div className="bg-gray-200 rounded-lg overflow-hidden">
                <div className="relative">
                  {isImageLoading ? (
                    <div className="w-full h-64 bg-gray-300 rounded-lg flex items-center justify-center">
                      <div className="flex items-center gap-2 text-gray-600">
                        <svg
                          className="animate-spin h-5 w-5"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        Loading image...
                      </div>
                    </div>
                  ) : (
                    <img
                      onClick={openModal}
                      src={accidentImageUrl || bgimage || "/placeholder.svg"}
                      alt="Accident Snapshot"
                      className="w-full h-64 object-cover cursor-pointer"
                      onError={(e) => {
                        console.log("Image failed to load, using fallback")
                        e.target.src = bgimage || "/placeholder.svg"
                      }}
                    />
                  )}
                </div>
                <p className="p-3 text-sm text-gray-600">
                  Frame captured at the moment of impact with object detection highlighting.
                </p>
              </div>
            </div>

            {/* Download Report Button */}
            <div className="bg-white/90 rounded-xl shadow-lg p-6 flex flex-col items-center justify-center">
              <button
                onClick={handleDownload}
                disabled={isGenerating}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-800 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 flex items-center gap-2"
              >
                {isGenerating ? (
                  <span className="flex items-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    GENERATING REPORT
                  </span>
                ) : (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    DOWNLOAD CONCISE REPORT
                  </>
                )}
              </button>
              <p className="mt-3 text-gray-600 text-sm text-center">
                Generate a professional PDF report with all incident details
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800/80 text-white p-4 text-center">
        <p>© 2025 Accident Detection System | All Rights Reserved</p>
      </footer>

      {/* Image Modal */}
      {isModalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeModal}
        >
          <div
            className="relative max-w-6xl max-h-full bg-white rounded-lg overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 z-10 bg-black bg-opacity-50 text-white hover:bg-opacity-70 transition-colors rounded-full p-2"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <div className="p-4 bg-gray-800 text-white">
              <h3 className="text-xl font-bold">Accident Snapshot - Full View</h3>
             
            </div>

            <div className="relative bg-gray-100 flex items-center justify-center min-h-[400px]">
              <img
                src={accidentImageUrl || bgimage || "/placeholder.svg"}
                alt="Accident Snapshot - Full View"
                className="max-w-full max-h-[70vh] object-contain"
                onError={(e) => {
                  console.log("Image failed to load, using fallback")
                  e.target.src = bgimage || "/placeholder.svg"
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AccidentDetailsPage