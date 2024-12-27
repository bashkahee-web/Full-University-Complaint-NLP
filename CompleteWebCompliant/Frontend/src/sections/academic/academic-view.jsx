import axios from 'axios';
import { useState } from 'react';
import toast from 'react-hot-toast';

import { LoadingButton } from '@mui/lab';
import { 
  Paper,
  Table, 
  Stack, 
  Button, 
  Dialog,
  TableRow, 
  TableBody,
  TableCell,
  TableHead,
  TextField,
  DialogTitle,
  DialogActions,
  DialogContent,
  TableContainer,
} from "@mui/material";

import Iconify from 'src/components/iconify';

import useComplaints from "../finance/useComplaints";

export default function AcademicView() {
    const { complaints,setComplaints, isLoading, error } = useComplaints('academic');
    const [reply, setReply] = useState([])
    const [stdID, setStdID] = useState(0)
    const [open, setOpen] = useState(false);
    const [activeComplaintId, setActiveComplaintId] = useState(null); // To track which complaint is being replied to
    

    const data = localStorage.getItem("userData" || {});
    const jsonData =JSON.parse(data);
    const isAdmin = jsonData.role.toLowerCase().trim() === "admin";

    if (isLoading) return <p>Loading...</p>;
    if (error) return <p>Error: {error}</p>;


    const handleOpen = (inboxId) => {
      setActiveComplaintId(inboxId);
      setOpen(true);
    };
  
    const handleClose = () => {
      setOpen(false);
      setReply('');
    };
  
    const handleSubmitResponse = async () => {
      if (!reply.trim()) {
        alert('Reply cannot be empty.');
        return;
      }
    
      try {
        const response = await axios.post('http://localhost:9000/api/responses', {
          text: reply,
          inboxId: jsonData._id , // Assuming jsonData contains the responder's ID
          responderId: stdID
        });
        console.log("Reply", response);
        const updatedComplaints = complaints.map(complaint =>
          complaint._id === activeComplaintId ? { ...complaint, response: reply } : complaint
        );
        setComplaints(updatedComplaints);
        handleClose();
      } catch (err) {
        console.error('Error submitting response:', err);
        alert('Failed to submit response.');
      }
    };
    

    const handleDeleteClick = async (id) => {
      try {
        const response = await axios.delete(`http://localhost:9000/api/delComplaint/${id}`);
        if (response.status === 200) {
          setComplaints((prevData) => prevData.filter((complaint) => complaint._id !== id));
          toast.success('Complaint deleted successfully',{position:"top-right"});
        } else {
          console.error('Delete failed:', response.data.message);
          toast.error('Failed to delete complaint',{position:"top-right"});
        }
      } catch (err) {
        console.error('Error deleting complaint:', err);
        toast.error('Error deleting complaint',{position:"top-right"});
      }
    };
    
  
  
    const handleComplete = async (id) => {
      try {
        const response = await axios.put(`http://localhost:9000/api/status/${id}`, {
          status: 'Completed',
        });
        setComplaints((prevData) =>
          prevData.map((item) =>
            item._id === id ? { ...item, status: response.data.status } : item
          )
        );
      } catch (err) {
        console.error('Error updating Compliant status:', err);
      }
    };
  
  
    return (
      <>
      <Dialog
          open={open}
          onClose={handleClose}
          aria-labelledby="alert-dialog-title"
          aria-describedby="alert-dialog-description"
          sx={{
            width: '100%', // Increases the width to 90% of the screen
          maxWidth: 'none', // Removes the maximum width constraint
          mx: 'auto',
                }}
          >
          <DialogTitle id="alert-dialog-title" sx={{marginBottom:"3px"}}>
            Make Response to Complaint<span style={{ color: 'white' }}>kkkkkkkkkkkk</span>
          </DialogTitle>
          <DialogContent>
            <Stack spacing={5}>
              <TextField
                name="description"
                label="Repy Description"
                multiline
                rows={3}
                placeholder="Enter a brief Reply of the page here"
                value={reply}
                onChange={(e) => setReply(e.target.value)}
                sx={{
                  width: '100%', // Ensures the TextField uses the full width of the dialog
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
            </Stack>
          </DialogContent>
          <DialogActions>
            <LoadingButton
            sx={{width:"92%", marginRight:"15px"}}
              // fullWidth
              size="large"
              type="submit"
              variant="contained"
           
              onClick={handleSubmitResponse}
            >
              Send
            </LoadingButton>
          </DialogActions>
        </Dialog>

      <div>
          <h1>Complaints in Academic</h1>
          <TableContainer component={Paper} sx={{ overflow: 'unset' }}>
            <Table sx={{ minWidth: 800 }}>
              <TableHead>
                <TableRow>
                  <TableCell>Description</TableCell> 
                  <TableCell>Status</TableCell> 
                  {!isAdmin && (
                    <TableCell>Reply</TableCell> 
                  )} 
                  {!isAdmin && (
                    <TableCell>Action</TableCell>
                  )}
                  {!isAdmin && (
                    <TableCell>Doing</TableCell>
                  )}
                </TableRow>
              </TableHead>
              <TableBody>
                {complaints.length > 0 ? complaints.map((complaint) => (
                  <TableRow key={complaint._id}>
                    <TableCell component="th" scope="row" sx={{ maxWidth: 250, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                      {complaint.description}
                    </TableCell>
                    <TableCell component="th" scope="row" sx={{fontWeight:"500",color: (complaint.status === 'Pending') ? 'red' : 'orange',}}>
                    {complaint.status}
                    </TableCell>
                    {!isAdmin && (
                    <TableCell>
                    <Button
                        variant="contained"
                        color="inherit"
                        startIcon={<Iconify icon="ic:round-reply" />}
                        onClick={() => {
                          handleOpen(complaint._id);
                          setStdID(complaint.stdId)
                        }}
                    >
                        Reply
                    </Button>
                    </TableCell>
                    )}
                    {!isAdmin && (
                    <TableCell>
                      <Button onClick={() => handleComplete(complaint._id)} sx={{
                        backgroundColor: 'green',color:"white",
                        '&:hover': {
                          backgroundColor: 'darkgreen'
                        }
                      }}>
                        Complete
                      </Button>
                    </TableCell>

                    )}
                    {!isAdmin && (
                    <TableCell component="th" scope="row" sx={{ maxWidth: 250, wordWrap: 'break-word', overflowWrap: 'break-word' }}>
                     <Iconify icon="ic:round-delete" sx={{ color: '#881337', marginLeft: 2, width: '24px', height: '24px' }} onClick={() => handleDeleteClick(complaint._id)}/>
                    </TableCell> 
                    )}
                  </TableRow>
                )) : <TableRow>
                  <TableCell style={{ textAlign: 'center' }} colSpan={6}>No complaints found.</TableCell>
                </TableRow>}
              </TableBody>
            </Table>
          </TableContainer>
      </div>
      </>
    );

}
