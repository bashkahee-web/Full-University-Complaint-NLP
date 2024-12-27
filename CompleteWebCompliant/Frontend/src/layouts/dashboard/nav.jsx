import PropTypes from 'prop-types';
import { useEffect, useContext} from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Drawer from '@mui/material/Drawer';
// import Button from '@mui/material/Button';
import Avatar from '@mui/material/Avatar';
import { alpha } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import ListItemButton from '@mui/material/ListItemButton';

import { usePathname } from 'src/routes/hooks';
import { RouterLink } from 'src/routes/components';

import { useResponsive } from 'src/hooks/use-responsive';

import { UserContext} from 'src/_mock/account';

import Logo from 'src/components/logo';
import Scrollbar from 'src/components/scrollbar';

import { NAV } from './config-layout';
import navConfig from './config-navigation';

// ----------------------------------------------------------------------

export default function Nav({ openNav, onCloseNav }) {
  const currentUser = useContext(UserContext);
  const pathname = usePathname();

  const upLg = useResponsive('up', 'lg');

  useEffect(() => {
    if (openNav) {
      onCloseNav();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

    // const data = localStorage.getItem("userData", JSON.stringify(currentUser));
    const data = localStorage.getItem("userData" || {});
    const jsonData =JSON.parse(data);
    // console.log("Hassan", jsonData.username)

  const renderAccount = (
    <Box
      sx={{
        my: 3,
        mx: 2.5,
        py: 2,
        px: 2.5,
        display: 'flex',
        borderRadius: 1.5,
        alignItems: 'center',
        bgcolor: (theme) => alpha(theme.palette.grey[500], 0.12),
      }}
    >
      <Avatar src={currentUser.photoURL} alt="photoURL" />

      <Box sx={{ ml: 2 }}>
        <Typography variant="subtitle2">{jsonData?.username}</Typography>

        {/* <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          {currentUser.email}
        </Typography> */}
      </Box>
    </Box>
  );


  const renderMenu = (
    <Stack component="nav" spacing={0.5} sx={{ px: 2 }}>
      {navConfig.map((item) => {
        const isUser = jsonData.role.toLowerCase().trim() === "user";
        const isAdmin = jsonData.role.toLowerCase().trim() === "admin";
        const isAcademicadmin = jsonData.role.toLowerCase().trim() === "acadmin";
        const isFinanceAdmin = jsonData.role.toLowerCase().trim() === "fcadmin";
        const isEqAdmin = jsonData.role.toLowerCase().trim() === "eqadmin";
        
        // If user, show only submit menu
        if (isUser && (item.path === '/submit' || item.path === '/studentinfo' || item.path === '/inbox')) {
          return <NavItem key={item.title} item={item} />;
        }

        if (isAcademicadmin && (item.path === '/academic')) {
          return <NavItem key={item.title} item={item} />;
        }
        if (isFinanceAdmin && (item.path === '/finance')) {
          return <NavItem key={item.title} item={item} />;
        }
        if (isEqAdmin && (item.path === '/equipment')) {
          return <NavItem key={item.title} item={item} />;
        }
      

        // If admin, show all except submit menu
        if (isAdmin) {
          if (item.path !== '/submit' && item.path !== '/studentinfo' && item.path !== '/inbox') {
            return <NavItem key={item.title} item={item} />;
          }
          return null;
        }
        
        // If none of the above conditions are met, do not render the item
        return null;
      })}
    </Stack>
  );

  const renderContent = (
    <Scrollbar
      sx={{
        height: 1,
        '& .simplebar-content': {
          height: 1,
          display: 'flex',
          flexDirection: 'column',
        },
      }}
    >
      <Logo sx={{ mt: 3, ml: 4 }} />

      {renderAccount}

      {renderMenu}

      <Box sx={{ flexGrow: 1 }} />

      {/* {renderUpgrade} */}
    </Scrollbar>
  );

  return (
    <Box
      sx={{
        flexShrink: { lg: 0 },
        width: { lg: NAV.WIDTH },
      }}
    >
      {upLg ? (
        <Box
          sx={{
            height: 1,
            position: 'fixed',
            width: NAV.WIDTH,
            borderRight: (theme) => `dashed 1px ${theme.palette.divider}`,
          }}
        >
          {renderContent}
        </Box>
      ) : (
        <Drawer
          open={openNav}
          onClose={onCloseNav}
          PaperProps={{
            sx: {
              width: NAV.WIDTH,
            },
          }}
        >
          {renderContent}
        </Drawer>
      )}
    </Box>
  );
}

Nav.propTypes = {
  openNav: PropTypes.bool,
  onCloseNav: PropTypes.func,
};

// ----------------------------------------------------------------------

function NavItem({ item }) {
  const pathname = usePathname();

  const active = item.path === pathname;

  return (
    <ListItemButton
      component={RouterLink}
      href={item.path}
      sx={{
        minHeight: 44,
        borderRadius: 0.75,
        typography: 'body2',
        color: 'text.secondary',
        textTransform: 'capitalize',
        fontWeight: 'fontWeightMedium',
        ...(active && {
          color: 'primary.main',
          fontWeight: 'fontWeightSemiBold',
          bgcolor: (theme) => alpha(theme.palette.primary.main, 0.08),
          '&:hover': {
            bgcolor: (theme) => alpha(theme.palette.primary.main, 0.16),
          },
        }),
      }}
    >
      <Box component="span" sx={{ width: 24, height: 24, mr: 2 }}>
        {item.icon}
      </Box>

      <Box component="span">{item.title} </Box>
    </ListItemButton>
  );
}

NavItem.propTypes = {
  item: PropTypes.object,
};
