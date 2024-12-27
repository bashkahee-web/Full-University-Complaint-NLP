import axios from 'axios';
import toast from 'react-hot-toast';
import React, { useState, useEffect } from 'react';

import { Paper, Table, TableRow, TableBody, TableCell, TableHead, TableContainer } from "@mui/material";

import Iconify from 'src/components/iconify';

const StudentData = () => {
  const [StdData, setstdData] = useState([]);
  console.log(StdData);

  useEffect(() => {
    const fetchStudentData = async () => {
      try {
        const jsonData = JSON.parse(localStorage.getItem('userData'));
        if (!jsonData || !jsonData._id) {
          throw new Error('Invalid user data in localStorage');
        }

        const stdId = jsonData._id;
        const response = await axios.get(`http://localhost:9000/api/studentData/${stdId}`);
        setstdData(response.data);
        console.log("here we go", response.data);
      } catch (error) {
        console.error('Error fetching Student data:', error);
      }
    };
    fetchStudentData();
  }, []);

  const handleDeleteClick = async (id) => {
    try {
      const response = await axios.delete(`http://localhost:9000/api/delComplaint/${id}`);
      if (response.status === 200) {
        setstdData((prevData) => prevData.filter((student) => student._id !== id));
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
      <h1>Complaints in Student</h1>
      <TableContainer component={Paper} sx={{ overflow: 'unset' }}>
        <Table sx={{ minWidth: 800 }}>
          <TableHead>
            <TableRow >
              <TableCell >Description</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {StdData.length > 0 ? StdData.map((student) => (
              <TableRow key={student._id}>
                <TableCell component="th" scope="row" sx={{ maxWidth: 150, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                  {student.description}
                </TableCell>
                <TableCell component="th" scope="row" sx={{ fontWeight: "500", color: (student.status === 'Pending') ? 'red' : 'green' }}>
                  {student.status}
                </TableCell>
                <TableCell component="th" scope="row" sx={{ maxWidth: 250, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                    <Iconify icon="ic:round-delete" sx={{ color: '#881337', marginLeft: 2, width: '25px', height: '25px' }} onClick={() => handleDeleteClick(student._id)}/>
                    {/* <Iconify icon="ic:baseline-edit" sx={{ color: 'blue' }}/> */}
                </TableCell>
              </TableRow>
            )) : <TableRow>
              <TableCell style={{ textAlign: 'center' }} colSpan={2}>No Student complaints found.</TableCell>
            </TableRow>}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
};

export default StudentData;
