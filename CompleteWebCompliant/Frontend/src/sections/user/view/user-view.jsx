import axios from 'axios';
import toast from 'react-hot-toast';
import { useState, useEffect } from 'react';
// import { useNavigate } from 'react-router-dom';

import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import { LoadingButton } from '@mui/lab';
import Dialog from '@mui/material/Dialog';
import Container from '@mui/material/Container';
import TableBody from '@mui/material/TableBody';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import { Button, Select,MenuItem,TextField, InputLabel, IconButton,FormControl, InputAdornment } from '@mui/material';
// import DialogContentText from '@mui/material/DialogContentText';

import { fetchRealUsers } from 'src/_mock/user';

import Iconify from 'src/components/iconify';
import Scrollbar from 'src/components/scrollbar';

import TableNoData from '../table-no-data';
import UserTableRow from '../user-table-row';
import UserTableHead from '../user-table-head';
import TableEmptyRows from '../table-empty-rows';
import UserTableToolbar from '../user-table-toolbar';
import { emptyRows, applyFilter, getComparator } from '../utils';


// ----------------------------------------------------------------------

export default function UserPage() {
  const [users, setUsers] = useState([]);
  // const navigate = useNavigate();  // Hook for navigation


  useEffect(() => {
    const loadUsers = async () => {
      const realUsers = await fetchRealUsers();
      setUsers(realUsers);
    };

    loadUsers();
  }, []);
  const [page, setPage] = useState(0);

  const [order, setOrder] = useState('asc');

  const [selected, setSelected] = useState([]);

  const [orderBy, setOrderBy] = useState('name');

  const [filterName, setFilterName] = useState('');

  const [rowsPerPage, setRowsPerPage] = useState(5);

  const handleSort = (event, id) => {
    const isAsc = orderBy === id && order === 'asc';
    if (id !== '') {
      setOrder(isAsc ? 'desc' : 'asc');
      setOrderBy(id);
    }
  };
  const [open, setOpen] = useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);
  const handleSelectAllClick = (event) => {
    if (event.target.checked) {
      const newSelecteds = users.map((n) => n.name);
      setSelected(newSelecteds);
      return;
    }
    setSelected([]);
  };

  const handleClick = (event, name) => {
    const selectedIndex = selected.indexOf(name);
    let newSelected = [];
    if (selectedIndex === -1) {
      newSelected = newSelected.concat(selected, name);
    } else if (selectedIndex === 0) {
      newSelected = newSelected.concat(selected.slice(1));
    } else if (selectedIndex === selected.length - 1) {
      newSelected = newSelected.concat(selected.slice(0, -1));
    } else if (selectedIndex > 0) {
      newSelected = newSelected.concat(
        selected.slice(0, selectedIndex),
        selected.slice(selectedIndex + 1)
      );
    }
    setSelected(newSelected);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setPage(0);
    setRowsPerPage(parseInt(event.target.value, 10));
  };

  const handleFilterByName = (event) => {
    setPage(0);
    setFilterName(event.target.value);
  };

  const dataFiltered = applyFilter({
    inputData: users,
    comparator: getComparator(order, orderBy),
    filterName,
  });

  const notFound = !dataFiltered.length && !!filterName;


  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    role: '',
});

const [showPassword, setShowPassword] = useState(false);

const handleChange = (event) => {
    const { name, value } = event.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
};

const handleSub = (event) => {
  event.preventDefault();

  // Correct the formData structure if necessary
  axios.post('http://localhost:9000/api/signup', formData)
  .then(result => {
      console.log("the result, ", result);
      handleClose(); // Close the modal on successful submission
      toast.success('User registered successfull',{position:"top-right"});
  })
  .catch(err => console.log(`The error, ${err}`));
};
  return (
  <>
     <Dialog
        open={open}
        onClose={handleClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
        // sx={{ width: '600px', maxWidth: '150%', mx: 'auto' }} 
      >
        <DialogTitle id="alert-dialog-title">
          Create An Account
        </DialogTitle>
        <DialogContent>
        <Stack spacing={3}>
        <TextField 
          name="name"  
          label="Your FullName" 
          value={formData.name}
          onChange={handleChange}
          sx={{ width: '350px', maxWidth: '150%', mx: 'auto' }} 
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
            <MenuItem value="acadmin">Ac Admin</MenuItem>
            <MenuItem value="fcadmin">Finance Admin</MenuItem>
            <MenuItem value="eqadmin">Eq Admin</MenuItem>
           </Select>
      </FormControl>
      </Stack>
        </DialogContent>
        <DialogActions>
        <LoadingButton
        fullWidth
        size="large"
        type="submit"
        variant="contained"
        color="inherit"
        onClick={handleSub}
      >
        SignUp
      </LoadingButton>
        </DialogActions>
      </Dialog>
    <Container>

      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={5}>
        <Typography variant="h4">Users</Typography>
          <Button
            variant="contained"
            color="inherit"
            startIcon={<Iconify icon="eva:plus-fill" />}
            onClick={handleOpen}  // Attach the navigation function to the onClick event
        >
            New User
        </Button>
      </Stack>

      <Card>
        <UserTableToolbar
          numSelected={selected.length}
          filterName={filterName}
          onFilterName={handleFilterByName}
        />
        <Scrollbar>
          <TableContainer sx={{ overflow: 'unset' }}>
            <Table sx={{ minWidth: 800 }}>
              <UserTableHead
                order={order}
                orderBy={orderBy}
                rowCount={users.length}
                numSelected={selected.length}
                onRequestSort={handleSort}
                onSelectAllClick={handleSelectAllClick}
                headLabel={[
                  { id: 'name', label: 'Name' },
                  { id: 'role', label: 'Role' },
                  { id: 'email', label: 'Email' },
                ]}
              />
              <TableBody>
                {dataFiltered
                  .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((row) => (
                    <UserTableRow
                      key={row.id}
                      name={row.name}
                      email={row.email}
                      role={row.role}
                      avatarUrl={row.avatarUrl}
                      selected={selected.indexOf(row.name) !== -1}
                      handleClick={(event) => handleClick(event, row.name)}
                    />
                  ))}

                <TableEmptyRows
                  height={77}
                  emptyRows={emptyRows(page, rowsPerPage, users.length)}
                />

                {notFound && <TableNoData query={filterName} />}
              </TableBody>
            </Table>
          </TableContainer>
        </Scrollbar>

        <TablePagination
          page={page}
          component="div"
          count={users.length}
          rowsPerPage={rowsPerPage}
          onPageChange={handleChangePage}
          rowsPerPageOptions={[5, 10, 25]}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Card>
    </Container>
  </>
  );
}
