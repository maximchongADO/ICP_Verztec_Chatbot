document.addEventListener("DOMContentLoaded", function () {
  const loginForm = document.getElementById("loginForm");
  const emailOrUsernameInput = document.getElementById("emailOrUsername");
  const passwordInput = document.getElementById("password");

  // Check if already logged in
  const token = localStorage.getItem("token");
  if (token) {
    window.location.href = "/chatbot.html";
    return;
  }
  loginForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const emailOrUsername = emailOrUsernameInput.value.trim();
    const password = passwordInput.value.trim();

    if (!emailOrUsername || !password) {
      showAlert("Please enter both email/username and password", "error");
      return;
    }

    // Show loading state
    const submitButton = loginForm.querySelector('button[type="submit"]');
    const loginText = submitButton.querySelector(".login-text");
    const loadingText = submitButton.querySelector(".loading-text");

    submitButton.disabled = true;
    loginText.style.display = "none";
    loadingText.style.display = "inline";

    try {
      // Determine if input is email or username
      const isEmail = emailOrUsername.includes("@");
      const requestBody = {
        password,
        ...(isEmail
          ? { email: emailOrUsername }
          : { username: emailOrUsername }),
      };

      const response = await fetch("/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Store token and user info
        localStorage.setItem("token", data.token);
        localStorage.setItem("userId", data.userId);

        showAlert("Login successful! Redirecting...", "success");

        // Redirect to chatbot page after short delay
        setTimeout(() => {
          window.location.href = "/chatbot.html";
        }, 1000);
      } else {
        showAlert(data.message || "Login failed. Please try again.", "error");
      }
    } catch (error) {
      console.error("Login error:", error);
      showAlert("Connection error. Please try again.", "error");
    } finally {
      // Reset button state
      submitButton.disabled = false;
      loginText.style.display = "inline";
      loadingText.style.display = "none";
    }
  });
});

function showAlert(message, type) {
  // Remove existing alerts
  const existingAlert = document.querySelector(".alert");
  if (existingAlert) {
    existingAlert.remove();
  }

  // Create new alert
  const alert = document.createElement("div");
  alert.className = `alert alert-${type}`;
  alert.textContent = message;

  // Insert alert before the form
  const form = document.getElementById("loginForm");
  form.parentNode.insertBefore(alert, form);

  // Auto-remove alert after 5 seconds
  setTimeout(() => {
    if (alert.parentNode) {
      alert.remove();
    }
  }, 5000);
}
