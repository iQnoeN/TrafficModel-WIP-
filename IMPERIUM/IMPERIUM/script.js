const express = require('express');
const app = express();
const PORT = 3000;

// Middleware to parse JSON
app.use(express.json());

// Define the endpoint to receive location data
app.post('/predict', async (req, res) => {
  try {
    // Extract location data from the request body
    const location = req.body.location;
    console.log(`Received location: ${location}`);

    // Replace this with your actual prediction logic
    // You could:
    // - Fetch data from a database (using Mongoose or another library)
    // - Call an external API for real-time traffic data
    // - Implement a machine learning model for traffic prediction
    const predictionData = await getPredictionData(location); // Replace with your logic

    const response = {
      message: `Prediction data for ${location} is available.`,
      location,
      predictionData, // Include the actual prediction data
    };

    res.status(200).json(response);
  } catch (error) {
    console.error('Error processing prediction request:', error);
    res.status(500).json({ error: 'Failed to generate prediction data' });
  }
});

// Function to simulate fetching prediction data (replace with your implementation)
async function getPredictionData(location) {
  // Replace this with logic to retrieve or generate prediction data
  // based on the location
  return {
    traffic_status: 'moderate', // Example prediction
    timestamp: new Date(),
  };
}

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});