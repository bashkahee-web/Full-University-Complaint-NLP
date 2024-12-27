import axios from 'axios';
import toast from 'react-hot-toast';
import React, { useState, useEffect } from 'react';

import { Paper, Table, TableRow, TableBody, TableCell, TableHead, TableContainer } from "@mui/material";

import Iconify from 'src/components/iconify';


const Inbox = () => {
  const [stdInbox, setstdInbox] = useState([]);

  useEffect(() => {
    const fetchStudentInbox = async () => {
      try {
        const jsonData = JSON.parse(localStorage.getItem('userData'));
        if (!jsonData || !jsonData._id) {
          throw new Error('Invalid user data in localStorage');
        }

        const UsertId = jsonData._id;
        console.log("Hasssssan",UsertId)
        const response = await axios.get(`http://localhost:9000/api/responses/${UsertId}`);
        setstdInbox(response.data);
        console.log("here we go", response.data);
      } catch (error) {
        console.error('Error fetching Student data:', error);
      }
    };
    fetchStudentInbox();
  }, []);


  const handleDeleteClick = async (id) => {
    try {
      const response = await axios.delete(`http://localhost:9000/api/delResponses/${id}`);
      if (response.status === 200) {
        setstdInbox((prevData) => prevData.filter((student) => student._id !== id));
        toast.success('Complaint deleted successfully',{position:"top-right"});
      } else {
        console.error('Delete failed:', response.data.message);
        toast.error('Failed to delete complaint',{position:"top-right"});
      }
    } catch (error) {
      console.error('Error deleting complaint:', error);
      toast.error('Error deleting complaint',{position:"top-right"});
    }
  };




  return (
    <div>
      <h1>Responses From stakeholders</h1>
      <TableContainer component={Paper} sx={{ overflow: 'unset' }}>
        <Table sx={{ minWidth: 800 }}>
          <TableHead>
            <TableRow >
              <TableCell >Response</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {stdInbox.length > 0 ? stdInbox.map((reply) => (
              <TableRow key={reply._id}>
                <TableCell component="th" scope="row" sx={{ maxWidth: 150, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                  {reply.text}
                </TableCell>
                <TableCell component="th" scope="row" sx={{ maxWidth: 250, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                    <Iconify icon="ic:round-delete" sx={{ color: '#881337', marginLeft: 2, width: '25px', height: '25px' }} onClick={() => handleDeleteClick(reply._id)}/>
                </TableCell>
              </TableRow>
            )) : <TableRow>
              <TableCell style={{ textAlign: 'center' }} colSpan={2}>No Student Inbox found.</TableCell>
            </TableRow>}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
};

export default Inbox;
