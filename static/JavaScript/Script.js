function showAlert(message = "Something happened!", timeout = 3000) {
    // Select elements correctly
    const alertBox = document.getElementById("customAlert");
    const alertMessage = document.getElementById("alertMessage");

    if (!alertBox || !alertMessage) {
        alert("❌ Error: Alert elements not found! Make sure 'customAlert' exists in your HTML.");
        return;
    }

    alertMessage.innerText = message; // Set message
    alertBox.style.display = "block"; // Show alert

    // Auto-hide alert after timeout
    setTimeout(() => {
        closeAlert();
    }, timeout);
}

function closeAlert() {
    const alertBox = document.getElementById("customAlert");
    if (alertBox) {
        alertBox.style.display = "none";
    }
}
