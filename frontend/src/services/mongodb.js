import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5000';

// Fetch all accidents with optional sorting
export const getAllAccidents = async (sortBySeverity = false) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/accidents`, {
      params: { sortBySeverity }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching accidents:', error);
    throw error;
  }
};

// Fetch a single accident by ID
export const getAccidentById = async (id) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/accidents/${id}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching accident details:', error);
    throw error;
  }
};

// Save accident data (this is called from useVideoUpload.js)
export const saveAccidentData = async (accidentData) => {
  // This function is not needed as the backend saves data automatically
  // But we keep it for potential future use
  console.log('Accident data saved via backend');
  return accidentData;
};