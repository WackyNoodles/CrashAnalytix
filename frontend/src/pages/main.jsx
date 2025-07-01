// Add this import
import { useNavigate } from "react-router-dom";

// Inside your component
function Home() {
  // Add this line
  const navigate = useNavigate();
  
  // Add this function
  const handleViewHistory = () => {
    navigate("/accident-history");
  };

  // In your JSX, add a button like:
  <button
    onClick={handleViewHistory}
    className="px-4 py-2 bg-white text-red-800 rounded-lg hover:bg-gray-100 transition-colors"
  >
    View Accident History
  </button>
}