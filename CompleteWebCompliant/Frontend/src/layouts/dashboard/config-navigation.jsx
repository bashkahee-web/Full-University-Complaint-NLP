import SvgColor from 'src/components/svg-color';

// ----------------------------------------------------------------------

const icon = (name) => (
  <SvgColor src={`/assets/icons/navbar/${name}.svg`} sx={{ width: 1, height: 1 }} />
);

const navConfig = [
  {
    title: 'dashboard',
    path: '/dashboard',
    icon: icon('ic_analytics'),
  },
  {
    title: 'Submittion',
    path: '/submit',
    icon: icon('ic_user'),
  },
  {
    title: 'Academic',
    path: '/academic',
    icon: icon('education'),
  },
  {
    title: 'Finance',
    path: '/finance',
    icon: icon('money'),
  },
  {
    title: 'Equipment',
    path: '/equipment',
    icon: icon('equip'),
  },
  {
    title: 'Studentinfo',
    path: '/studentinfo',
    icon: icon('ic_analytics'),
  },
  {
    title: 'Inbox',
    path: '/inbox',
    icon: icon('ic_analytics'),
  },
  // {
  //   title: (
  //     <div style={{ display: 'flex', alignItems: 'center' }}>
  //       <span>Inbox</span>
  //       <Badge showZero badgeContent={unreadMessages} color="error" max={99} style={{ marginLeft: 123 }} />
  //     </div>
  //   ),
  //   path: '/inbox',
  //   icon: icon('ic_analytics'),
  // },
  {
    title: 'user',
    path: '/user',
    icon: icon('ic_user'),
  },
  {
    title: 'login',
    path: '/',
    icon: icon('ic_lock'),
  },
  // {
  //   title: 'Not found',
  //   path: '/404',
  //   icon: icon('ic_disabled'),
  // },
];

export default navConfig;
