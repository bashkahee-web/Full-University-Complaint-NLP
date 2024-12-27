import axios from 'axios';

export const fetchRealUsers = async () => {
  try {
    const response = await axios.get('http://localhost:9000/api/getusers');
    const fetchedUsers = response.data.User; // Adjust this according to your actual API response structure
    return fetchedUsers.map((user, index) => ({
      id: user._id, // Assuming user data includes an _id field
      avatarUrl: `/assets/images/avatars/avatar_${index % 24 + 1}.jpg`, // Cycling through 24 avatars
      name: user.name, // Using the real name from the database
      email: user.email || 'Unknown email',
      role: user.role
    }));
  } catch (error) {
    console.error('Failed to fetch users:', error);
    return []; // Return an empty array on error
  }
};

