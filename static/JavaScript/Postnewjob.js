

validateToken()
async function postJob(event) {
    event.preventDefault();
    //Retrieves the values entered by the user in the form fields using their id.
    const jobTitle = document.getElementById('Job_Title').value;
    const jobDescription = document.getElementById('Job_Description').value;
    const experience = document.getElementById('Experience').value;
    const skills = document.getElementById('Skills').value;
    const location = document.getElementById('Location').value;
    //Create a data object
    var object = {
        "Job_Title": jobTitle,
        "Job_Description": jobDescription,
        "Experience": experience,
        "Skills": skills,
        "Location": location

    }

    try {
        // Send data to the backend
        const response = await fetch("/postnewjob", {
            method: "POST",
            headers: {//Headers indicate that the request body is in JSON format.
                "Content-Type": "application/json",  // Set header for JSON
            },
            body: JSON.stringify(object),  // Convert object to JSON string
        });

        const result = await response.json();
        if (response.ok) {
            alert("job posted successfully")
            // showAlert("Job posted successfully!");
            window.location.href = "/joblist"
        } else {
            alert(result.msg || "something went wrong"); 
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong! Please try again.");
    }

}

async function validateToken() {
    const token = localStorage.getItem("access_token");
    if (!token) {
        alert("Unauthorized access. Please log in.");
        window.location.href = "/login";

    }
}


