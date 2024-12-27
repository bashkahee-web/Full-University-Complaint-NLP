import axios from 'axios';
import toast from 'react-hot-toast';
import React, { useState } from 'react';

import { Stack, Button, TextField } from '@mui/material';

export default function SubmitView() {
  // State for the description text, category result, and any error messages
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [error, setError] = useState('');

    // Function to handle submission
    const handleSubmit = async () => {
      const jsonData = JSON.parse(localStorage.getItem('userData'));
      const stdId = jsonData._id;

      // Validate the description field
    if (!description) {
      setError('Description cannot be empty.');
      toast.error("Description cannot be empty.",{position:"top-right"});
      return;
    }

      try {
        // Send the description to the Flask API for classification
        const classificationResponse = await axios.post('http://192.168.59.36:5000/predict', { text: description }, {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        });

        // Set the category from the Flask API response
        setCategory(classificationResponse.data.category);
        
        // Construct payload with the returned category
        const payload = {
          description,
          category: classificationResponse.data.category,
          stdId
        };
  
        // Send the description and category to the Node.js backend to be saved in MongoDB
        await axios.post('http://localhost:9000/api/complaints', payload, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
  
        // Optionally reset description and show success message
        setDescription('');
        setError('');
  
      } catch (err) {
        setError('Failed to process or save data. Please check the servers and try again.');
        console.error('Error:', err.response || err);
      }
    };


  return (
    <Stack spacing={3} direction="column" alignItems="center">
      {/* Page Description Field */}
      <TextField
        name="description"
        label="Page Description"
        multiline
        rows={5}
        placeholder="Enter a brief description of the page here"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        sx={{
          width: '60%',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.06)',
          '& .MuiOutlinedInput-root': {
            '&:hover fieldset': {
              borderColor: 'primary.main',
            },
            '&.Mui-focused fieldset': {
              borderColor: 'primary.main',
            }
          }
        }}
      />

      {/* Submit Button */}
      <Button
        variant="contained"
        color="primary"
        sx={{ minWidth: '130px', mt: 1, height: '50px' }}
        onClick={handleSubmit}
      >
        Submit
      </Button>

      {/* Display category and error message */}
      {category && <p>Cabashadaada waxay ku saabsan tahay <span style={{ color: 'red', fontWeight: 'bold' }}>{category}</span> insha allah jawaab ayaan ku soo siin doonaa</p>}
      {error && <p>Error: {error}</p>}
    </Stack>
  );
}
