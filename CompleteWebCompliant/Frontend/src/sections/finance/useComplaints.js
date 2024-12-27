// src/hooks/useComplaints.js
import axios from 'axios';
import { useState, useEffect } from 'react';

function useComplaints(category) {
    const [complaints, setComplaints] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        if (!category) {
            setError('Category is undefined or not provided');
            setIsLoading(false);
            return;
        }

        setIsLoading(true);
        axios.get(`http://localhost:9000/api/complaints/${category}`)
            .then(response => {
                setComplaints(response.data);
                setIsLoading(false);
            })
            .catch(err => {
                console.error("Error fetching data:", err);
                setError('Unable to fetch data.');
                setIsLoading(false);
            });
    }, [category]);  // Dependency on category ensures this runs only when category changes

    return { complaints, isLoading, error, setComplaints };
}

export default useComplaints;
