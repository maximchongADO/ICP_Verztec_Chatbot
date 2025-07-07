// Simple test to check if current token is valid
const token = localStorage.getItem("token");
console.log('Current token:', token);

if (token) {
  fetch('http://localhost:8000/api/users/me', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
  .then(response => {
    console.log('Token test response status:', response.status);
    return response.json();
  })
  .then(data => {
    console.log('Token test response data:', data);
  })
  .catch(error => {
    console.error('Token test failed:', error);
  });
} else {
  console.log('No token found in localStorage');
}
