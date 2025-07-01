
// import jsPDF from "jspdf";
// import { useState } from 'react';
// import accidentImage from "./assets/dummy.png";

// const ProfessionalReport = () => {
//   const [isGenerating, setIsGenerating] = useState(false);

//   const generatePDF = () => {
//     const doc = new jsPDF();
//     const pageWidth = doc.internal.pageSize.getWidth();
//     const margin = 20;
//     let y = 15;

//     // ===== HEADER =====
//     doc.setTextColor(0, 51, 102); // Darker blue for better contrast
//     doc.setFontSize(20);
//     doc.setFont("helvetica", "bold");
//     doc.text("CrashAnalytix Accident Report", pageWidth / 2, y, { align: "center" });
//     y += 12;

//     doc.setDrawColor(180, 180, 180); // Slightly darker divider
//     doc.line(margin, y, pageWidth - margin, y);
//     y += 8;

//     // ===== REPORT SUMMARY =====
//     const summaryBoxHeight = 22;
//     doc.setDrawColor(210); // Slightly darker border
//     doc.setFillColor(245, 245, 245); // Slightly darker background
//     doc.roundedRect(margin, y, pageWidth - 2 * margin, summaryBoxHeight, 4, 4, 'FD');

//     doc.setFontSize(12);
//     doc.setFont("helvetica", "bold");
//     doc.setTextColor(0, 51, 102); // Matching header color
//     doc.text("REPORT SUMMARY", margin + 5, y + 7);

//     doc.setFontSize(10);
//     doc.setFont("helvetica", "normal");
//     doc.setTextColor(70, 70, 70); // Darker gray for better readability
//     doc.text("Timestamp: 14:25", margin + 5, y + 13);

//     y += summaryBoxHeight + 8;

//     // ===== SEVERITY ASSESSMENT =====
//     doc.setFontSize(12);
//     doc.setFont("helvetica", "bold");
//     doc.setTextColor(0, 51, 102); // Matching header color
//     doc.text("SEVERITY ASSESSMENT", margin, y);
//     y += 8;

//     doc.setFontSize(36);
//     doc.setFont("helvetica", "bold");
//     doc.setTextColor(204, 0, 0); // Red for severity score
//     doc.text("8.7", pageWidth / 2, y, { align: "center" });
//     y += 12;

//     const riskBarWidth = pageWidth - 2 * margin;
//     const riskBarHeight = 8;

//     // Enhanced gradient colors
//     doc.setFillColor(0, 153, 51); // Darker green
//     doc.rect(margin, y, riskBarWidth * 0.3, riskBarHeight, 'F');
//     doc.setFillColor(255, 204, 0); // Brighter yellow
//     doc.rect(margin + riskBarWidth * 0.3, y, riskBarWidth * 0.3, riskBarHeight, 'F');
//     doc.setFillColor(204, 0, 0); // Darker red
//     doc.rect(margin + riskBarWidth * 0.6, y, riskBarWidth * 0.4, riskBarHeight, 'F');

//     doc.setFillColor(51, 51, 51); // Darker marker
//     doc.rect(margin + riskBarWidth * 0.87 - 1, y - 2, 2, riskBarHeight + 4, 'F');

//     y += riskBarHeight + 5;

//     doc.setFontSize(10);
//     doc.setFont("helvetica", "normal");
//     doc.setTextColor(70, 70, 70);
//     doc.text("Low Risk", margin, y);
//     doc.text("High Risk", pageWidth - margin - 25, y, { align: "right" });
//     y += 10;

//     doc.setDrawColor(180, 180, 180);
//     doc.line(margin, y, pageWidth - margin, y);
//     y += 8;

//     // ===== INCIDENT DETAILS BOX =====
//     const tableBoxHeight = 45;
//     doc.setDrawColor(210);
//     doc.setFillColor(245, 245, 245);
//     doc.roundedRect(margin, y, pageWidth - 2 * margin, tableBoxHeight, 4, 4, 'FD');

//     doc.setFontSize(12);
//     doc.setFont("helvetica", "bold");
//     doc.setTextColor(0, 51, 102);
//     doc.text("INCIDENT DETAILS", margin + 5, y + 7);

//     y += 10;

//     // Table header with subtle background
//     doc.setFillColor(235, 235, 235);
//     doc.rect(margin, y, pageWidth - 2 * margin, 10, 'F');
//     doc.setFontSize(10);
//     doc.setFont("helvetica", "bold");
//     doc.setTextColor(0, 51, 102);
//     doc.text("Category", margin + 5, y + 7);
//     doc.text("Details", margin + 70, y + 7);
//     y += 8;

//     // Table rows with alternating subtle shading
//     const incidentDetails = [
//       { label: "Entities Involved", value: "2 Cars, 1 Pedestrian" },
//       { label: "Accident Type", value: "Rear-end, Side-impact" },
//       { label: "License Plates", value: "ABC-1234, XYZ-5678" }
//     ];

//     incidentDetails.forEach((item, index) => {
//         if (index % 2 === 0) {
//           doc.setFillColor(250, 250, 250);
//           doc.rect(margin, y, pageWidth - 2 * margin, 10, 'F');
//         }
      
//         doc.setFont("helvetica", "normal");
//         doc.setFontSize(9);
//         doc.setTextColor(...(index === 2 ? [204, 0, 0] : [70, 70, 70])); // Corrected
//         doc.text(item.label, margin + 5, y + 7);
//         doc.setTextColor(...(index === 2 ? [204, 0, 0] : [40, 40, 40])); // Corrected
//         doc.text(item.value, margin + 70, y + 7);
//         y += 8;
//       });
      

//     y += 10;

//     // ===== ACCIDENT IMAGE =====
//     doc.setFontSize(12);
//     doc.setFont("helvetica", "bold");
//     doc.setTextColor(0, 51, 102);
//     doc.text("ACCIDENT SNAPSHOT", margin, y);
//     y += 15;

//     const imgWidth = (pageWidth - 2 * margin) * 0.8;
//     const imgHeight = (imgWidth * 9) / 16;
//     const imgX = (pageWidth - imgWidth) / 2;

//     // Enhanced image border
//     const boxPadding = 4;
//     doc.setDrawColor(180, 180, 180);
//     doc.setLineWidth(0.8); // Slightly thicker border
//     doc.rect(imgX - boxPadding, y - boxPadding, imgWidth + 2 * boxPadding, imgHeight + 2 * boxPadding);

//     doc.addImage(accidentImage, "PNG", imgX, y, imgWidth, imgHeight);
//     y += imgHeight + boxPadding * 2 + 5;

//     // ===== FOOTER =====
//     doc.setDrawColor(180, 180, 180);
//     doc.setLineWidth(0.5);
//     doc.line(20, y, 190, y);

//     doc.setFontSize(10);
//     doc.setTextColor(100, 100, 100);
//     doc.text("Generated by | CrashAnalytiX", 105, y + 10, {
//       align: "center",
//     });

//     return doc;
//   };

//   const handleDownload = () => {
//     setIsGenerating(true);
//     setTimeout(() => {
//       try {
//         const doc = generatePDF();
//         doc.save(`crashanalytix_report_${new Date().getTime()}.pdf`);
//       } catch (error) {
//         console.error("PDF generation error:", error);
//       } finally {
//         setIsGenerating(false);
//       }
//     }, 300);
//   };

//   return (
//     <div className="flex flex-col items-center p-6">
//       <button
//         onClick={handleDownload}
//         disabled={isGenerating}
//         className="px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-800 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
//       >
//         {isGenerating ? (
//           <span className="flex items-center gap-3">
//             <svg className="animate-spin h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
//               <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
//               <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
//             </svg>
//             GENERATING REPORT
//           </span>
//         ) : (
//           'DOWNLOAD REPORT (PDF)'
//         )}
//       </button>
//     </div>
//   );
// };

// export default ProfessionalReport;
import jsPDF from "jspdf";
import { useState } from 'react';
import accidentImage from "./assets/dummy.png";

const ProfessionalReport = ({ accidentData }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Use default data if no accidentData is provided
  const accidentDetails = accidentData || {
    severity: "moderate",
    vehicles: {
      count: 2,
      types: ["Sedan", "SUV"],
    },
    pedestrians: 1,
    licensePlate: "ABC-1234",
    visibleLicensePlates: ["ABC-1234", "XYZ-5678"],
    timestamp: new Date().toLocaleString(),
    accidentType: "Vehicle collision",
    classification: "Accident",
  };

  const generatePDF = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 20;
    let y = 15;

    // ===== HEADER =====
    doc.setTextColor(0, 51, 102); // Darker blue for better contrast
    doc.setFontSize(20);
    doc.setFont("helvetica", "bold");
    doc.text("CrashAnalytix Accident Report", pageWidth / 2, y, { align: "center" });
    y += 12;

    doc.setDrawColor(180, 180, 180); // Slightly darker divider
    doc.line(margin, y, pageWidth - margin, y);
    y += 8;

    // ===== REPORT SUMMARY =====
    const summaryBoxHeight = 22;
    doc.setDrawColor(210); // Slightly darker border
    doc.setFillColor(245, 245, 245); // Slightly darker background
    doc.roundedRect(margin, y, pageWidth - 2 * margin, summaryBoxHeight, 4, 4, 'FD');

    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 51, 102); // Matching header color
    doc.text("REPORT SUMMARY", margin + 5, y + 7);

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(70, 70, 70); // Darker gray for better readability
    doc.text("Timestamp: " + accidentDetails.timestamp, margin + 5, y + 13);

    y += summaryBoxHeight + 8;

    // ===== SEVERITY ASSESSMENT =====
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 51, 102); // Matching header color
    doc.text("SEVERITY ASSESSMENT", margin, y);
    y += 8;

    // Set color based on severity
    const severityColors = {
      minor: [0, 153, 51],    // Green
      moderate: [255, 153, 0], // Orange
      severe: [204, 0, 0]      // Red
    };
    
    // Get color based on severity (default to moderate if not found)
    const textColor = severityColors[accidentDetails.severity] || severityColors.moderate;
    
    // Display severity text instead of number
    doc.setFontSize(28);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...textColor);
    
    // Capitalize first letter of severity
    const displaySeverity = accidentDetails.severity.charAt(0).toUpperCase() + 
                           accidentDetails.severity.slice(1);
    
    doc.text(displaySeverity, pageWidth / 2, y, { align: "center" });
    y += 12;

    const riskBarWidth = pageWidth - 2 * margin;
    const riskBarHeight = 8;

    // Enhanced gradient colors
    doc.setFillColor(0, 153, 51); // Darker green
    doc.rect(margin, y, riskBarWidth * 0.3, riskBarHeight, 'F');
    doc.setFillColor(255, 204, 0); // Brighter yellow
    doc.rect(margin + riskBarWidth * 0.3, y, riskBarWidth * 0.3, riskBarHeight, 'F');
    doc.setFillColor(204, 0, 0); // Darker red
    doc.rect(margin + riskBarWidth * 0.6, y, riskBarWidth * 0.4, riskBarHeight, 'F');

    // Position marker based on severity
    let markerPosition;
    switch (accidentDetails.severity) {
      case "minor":
        markerPosition = 0.15; // Left position (green zone)
        break;
      case "moderate":
        markerPosition = 0.45; // Middle position (yellow zone)
        break;
      case "severe":
        markerPosition = 0.8; // Right position (red zone)
        break;
      default:
        markerPosition = 0.45; // Default to middle if unknown
    }

    doc.setFillColor(51, 51, 51); // Darker marker
    doc.rect(margin + riskBarWidth * markerPosition - 1, y - 2, 2, riskBarHeight + 4, 'F');

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
    doc.roundedRect(margin, y, pageWidth - 2 * margin, tableBoxHeight, 4, 4, 'FD');

    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 51, 102);
    doc.text("INCIDENT DETAILS", margin + 5, y + 7);

    y += 10;

    // Table header with subtle background
    doc.setFillColor(235, 235, 235);
    doc.rect(margin, y, pageWidth - 2 * margin, 10, 'F');
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 51, 102);
    doc.text("Category", margin + 5, y + 7);
    doc.text("Details", margin + 70, y + 7);
    y += 8;

    // Table rows with alternating subtle shading
    const vehicleTypes = accidentDetails.vehicles.types.join(", ");
    const licensePlates = accidentDetails.visibleLicensePlates.join(", ");
    
    const incidentDetails = [
      { 
        label: "Entities Involved", 
        value: `${accidentDetails.vehicles.count} Vehicles, ${accidentDetails.pedestrians} Pedestrian` 
      },
      { 
        label: "Accident Type", 
        value: accidentDetails.accidentType 
      },
      { 
        label: "License Plates", 
        value: licensePlates 
      }
    ];

    incidentDetails.forEach((item, index) => {
      if (index % 2 === 0) {
        doc.setFillColor(250, 250, 250);
        doc.rect(margin, y, pageWidth - 2 * margin, 10, 'F');
      }
    
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(...(index === 2 ? [204, 0, 0] : [70, 70, 70])); // Corrected
      doc.text(item.label, margin + 5, y + 7);
      doc.setTextColor(...(index === 2 ? [204, 0, 0] : [40, 40, 40])); // Corrected
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

    // Enhanced image border
    const boxPadding = 4;
    doc.setDrawColor(180, 180, 180);
    doc.setLineWidth(0.8); // Slightly thicker border
    doc.rect(imgX - boxPadding, y - boxPadding, imgWidth + 2 * boxPadding, imgHeight + 2 * boxPadding);

    doc.addImage(accidentImage, "PNG", imgX, y, imgWidth, imgHeight);
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
    setIsGenerating(true);
    setTimeout(() => {
      try {
        const doc = generatePDF();
        doc.save(`crashanalytix_report_${new Date().getTime()}.pdf`);
      } catch (error) {
        console.error("PDF generation error:", error);
      } finally {
        setIsGenerating(false);
      }
    }, 300);
  };

  return (
    <div className="flex flex-col items-center p-6">
      <button
        onClick={handleDownload}
        disabled={isGenerating}
        className="px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-800 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
      >
        {isGenerating ? (
          <span className="flex items-center gap-3">
            <svg className="animate-spin h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            GENERATING REPORT
          </span>
        ) : (
          'DOWNLOAD REPORT (PDF)'
        )}
      </button>
    </div>
  );
};

export default ProfessionalReport;