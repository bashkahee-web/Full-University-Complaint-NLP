import axios from 'axios'
import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom';

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
import { Select, MenuItem,InputLabel, FormControl } from '@mui/material';

import { bgGradient } from 'src/theme/css';

import Logo from 'src/components/logo';
import Iconify from 'src/components/iconify';


// ----------------------------------------------------------------------

export default function SignUpView() {
  // const navigate =  useNavigate()
  const theme = useTheme();

  const navigate =  useNavigate();

  const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
        role: 'user',
    });

  const [showPassword, setShowPassword] = useState(false);

  const handleChange = (event) => {
        const { name, value } = event.target;
            setFormData(prevState => ({
                ...prevState,
                [name]: value
            }));
    };


 const handleClick = (event) => {
    event.preventDefault();

    // Correct the formData structure if necessary
    axios.post('http://localhost:9000/api/signup', formData)
    .then(result => {
        console.log("the result, ", result);
        navigate('/');
    })
    .catch(err => console.log(`The error, ${err}`));
};

  const renderForm = (
    <>
      <Stack spacing={3}>
        <TextField 
          name="name"  
          label="Your FullName" 
          value={formData.name}
          onChange={handleChange}
        />
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
        <FormControl fullWidth>
        <InputLabel id="role-label">Role</InputLabel>
          <Select
            labelId="role-label"
            name="role"
            value={formData.role}
            label="Role"
            onChange={handleChange}>
            <MenuItem value="user">User</MenuItem>
            <MenuItem value="admin">Admin</MenuItem>
           </Select>
      </FormControl>
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
        SignUp
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
          <Typography variant="h4">Sign in to Minimal</Typography>

          <Typography variant="body2" sx={{ mt: 2, mb: 5 }}>
            Have an account?
           
            <Link to='/login' >
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
