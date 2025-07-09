document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("signupForm");//Grabs the form with the id="signupForm" from the HTML.

    form.addEventListener("submit", async (event) => {//Adds an event listener for the form submission.
        event.preventDefault();

        const formData = new FormData(form);//Creates a FormData object with all the form’s inputs.

        try {
            const response = await fetch("/register", {
                method: "POST",
                body: formData,
            });
            // converts the response back into a usable js object
            const result = await response.json();
            if (response.ok) {
                alert(result.success_msg);
                window.location.href = "/login";  // Redirect after success
            } else {
                alert(result.msg);
            }
        } catch (error) {
            console.error("Error:", error);
        }
    });
});
