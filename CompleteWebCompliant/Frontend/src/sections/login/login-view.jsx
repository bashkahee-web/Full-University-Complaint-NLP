import axios from 'axios'
import toast from 'react-hot-toast';
import React, { useState, useEffect } from 'react' 
import { Link, useLocation, useNavigate } from 'react-router-dom';

import Box from '@mui/material/Box';
// import Link from '@mui/material/Link';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
// import Button from '@mui/material/Button';
// import Divider from '@mui/material/Divider';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import LoadingButton from '@mui/lab/LoadingButton';
import { alpha, useTheme } from '@mui/material/styles';
import InputAdornment from '@mui/material/InputAdornment';

import { bgGradient } from 'src/theme/css';

import Logo from 'src/components/logo';
import Iconify from 'src/components/iconify';


// ----------------------------------------------------------------------

export default function LoginView() {
  const navigate =  useNavigate();
  const { pathname } = useLocation(); // Destructure pathname from useLocation
  const theme = useTheme();

  const [formData, setFormData] = useState({
      email: '',
      password: ''
  });

  const [showPassword, setShowPassword] = useState(false);

      // Clear localStorage data when the user navigates to the login page
  useEffect(() => {
    if (pathname === '/') {
      // localStorage.clear();
      localStorage.removeItem('userData');
    }
  }, [pathname]); // Use the pathname variable instead of location.pathname

  const handleChange = (event) => {
        const { name, value } = event.target;
            setFormData(prevState => ({
                ...prevState,
                [name]: value
            }));
    };


  const handleClick = (event) => {
      event.preventDefault();
      const { email, password } = formData;
      axios.post('http://localhost:9000/api/loginuser', { email, password })
        .then(response => {
          if (response.data.message === "Login Successfull") {
            console.log(response);
            const jsonData ={
              "_id": response.data.user._id,
              "role": response.data.user.role,
              "username": response.data.user.name,
            }
            localStorage.setItem("userData",JSON.stringify(jsonData));
            if (jsonData.role === 'admin') {
              navigate('/dashboard'); // Admins go to the dashboard
            } else if (jsonData.role === 'acadmin') {
              navigate('/academic'); // Academic Admins go to the academic admin page
            } else if (jsonData.role === 'fcadmin') {
              navigate('/finance'); // Finance Admins go to the finance admin page
            } else if (jsonData.role === 'eqadmin') {
              navigate('/equipment'); // Equipment Admins go to the equipment admin page
            } else {
              navigate('/submit'); // Non-admins go to the submit menu
            }
          } else if (response.data === "Invalid Email please provide a valid email") {
            toast.error("Invalid Email please provide a valid email.", {position:"top-right"});
          } else if (response.data === "Incorrect Password") {
            toast.error("Incorrect Password.",{position:"top-right"});
          }
        })
        .catch(error => {
          toast.error("An error occurred.", {position:"top-right"});
        });
  };
      
      const renderForm = (
        <>
      <Stack spacing={3}>
        <TextField 
          name="email" 
          label="Email address" 
          value={formData.email}
          onChange={handleChange}
        />
        <TextField
          name="password"
          label="Password"
          type={showPassword ? 'text' : 'password'}
          value={formData.password}
          onChange={handleChange}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={() => setShowPassword(!showPassword)} edge="end">
                  <Iconify icon={showPassword ? 'eva:eye-fill' : 'eva:eye-off-fill'} />
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
      </Stack>

      <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ my: 3 }}>
        {/* <Link variant="subtitle2" underline="hover">
          Forgot password?
        </Link> */}
      </Stack>

      <LoadingButton
        fullWidth
        size="large"
        type="submit"
        variant="contained"
        color="inherit"
        onClick={handleClick}
      >
        Login
      </LoadingButton>
    </>
  );

  return (
    
    <Box
      sx={{
        ...bgGradient({
          color: alpha(theme.palette.background.default, 0.9),
          imgUrl: '/assets/background/overlay_4.jpg',
        }),
        height: 1,
      }}
    >
      <Logo
        sx={{
          position: 'fixed',
          top: { xs: 16, md: 24 },
          left: { xs: 16, md: 24 },
        }}
      />

      <Stack alignItems="center" justifyContent="center" sx={{ height: 1 }}>
        <Card
          sx={{
            p: 5,
            width: 1,
            maxWidth: 420,
          }}
        >
          <Typography variant="h4">Log in to Compliant</Typography>

          <Typography variant="body2" sx={{ mt: 2, mb: 5 }}>
            Don’t have an account?
           
            <Link to=''>
              Get started
            </Link>
          </Typography>
          <form onSubmit={handleClick}>
            {renderForm}
          </form>
        </Card>
      </Stack>
    </Box>
  );
}
