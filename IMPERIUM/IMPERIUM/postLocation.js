// Get reference to the dropdown
const locationDropdown = document.getElementById('location-dropdown');

// Define the function for posting the selected location
function postLocationName(locationName) {
    const url = 'http://localhost:3000'; // Replace with your server endpoint

    // Prepare the data to be sent
    const data = {
        location: locationName
    };

    // Make a POST request using Fetch API
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        // You can add more logic here to handle the response (e.g., display predictions)
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Event listener for when the dropdown value changes
locationDropdown.addEventListener('change', function () {
    const selectedOption = this.options[this.selectedIndex].text; // Get the selected location name
    if (selectedOption && selectedOption !== "Select a location...") {
        postLocationName(selectedOption);
    }
});