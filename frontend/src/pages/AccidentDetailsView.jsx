import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getAccidentById } from "../services/mongodb";
import AccidentDetailsPage from "../pages/AccidentDetailsPage"; // Your existing component

function AccidentDetailsView() {
  const [accidentData, setAccidentData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { id } = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    if (id) {
      fetchAccidentData(id);
    }
  }, [id]);

  const fetchAccidentData = async (accidentId) => {
  try {
    setLoading(true);
    const data = await getAccidentById(accidentId);
    if (data) {
      // Transform entities into frontend structure
      const vehicles = [];
      let pedestrianCount = 0;

      if (Array.isArray(data.entities)) {
        data.entities.forEach((e) => {
          if (e.type === "pedestrian") {
            pedestrianCount++;
          } else {
            vehicles.push(e.type);
          }
        });
      }

      const transformed = {
        ...data,
        vehicles: {
          count: vehicles.length,
          types: [...new Set(vehicles.map((v) => v.charAt(0).toUpperCase() + v.slice(1)))]
        },
        pedestrians: pedestrianCount,
        visibleLicensePlates: data.entities
          ?.filter((e) => e.license_plate)
          .map((e) => e.license_plate) || []
      };

      setAccidentData(transformed);
    } else {
      setError("Accident record not found");
    }
  } catch (err) {
    console.error("Failed to fetch accident details:", err);
    setError("Failed to load accident details. Please try again later.");
  } finally {
    setLoading(false);
  }
};


  const handleBack = () => {
    navigate("/accident-history");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex justify-center items-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-700 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading accident details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex justify-center items-center bg-gray-50">
        <div className="bg-white p-8 rounded-xl shadow-md max-w-md w-full text-center">
          <div className="inline-flex justify-center items-center w-16 h-16 bg-red-100 rounded-full mb-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-8 w-8 text-red-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={handleBack}
            className="px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors"
          >
            Back to History
          </button>
        </div>
      </div>
    );
  }

  return <AccidentDetailsPage onBack={handleBack} accidentData={accidentData} />;
}

export default AccidentDetailsView;