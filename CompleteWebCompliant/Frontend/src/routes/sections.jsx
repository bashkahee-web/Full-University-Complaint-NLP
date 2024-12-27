import { lazy, Suspense } from 'react';
import { Outlet, useRoutes } from 'react-router-dom';

import DashboardLayout from 'src/layouts/dashboard';


export const IndexPage = lazy(() => import('src/pages/app'));
// export const BlogPage = lazy(() => import('src/pages/blog'));
export const UserPage = lazy(() => import('src/pages/user'));
export const LoginPage = lazy(() => import('src/pages/login'));
export const SignUpPage = lazy(() => import('src/pages/SignUp'));
export const SubmittionPage = lazy(() => import('src/pages/submittion'));
export const AcademicPage = lazy(() => import('src/pages/academic'));
export const FinancePage = lazy(() => import('src/pages/finance'));
export const EquipmentPage = lazy(() => import('src/pages/equipment'));
export const StudentPage = lazy(() => import('src/pages/StudentInfo'));
export const StudentBox = lazy(() => import('src/pages/StudentInbox'));
export const Page404 = lazy(() => import('src/pages/page-not-found'));

// ----------------------------------------------------------------------

export default function Router() {
  const routes = useRoutes([
    {
      element: (
        <DashboardLayout>
          <Suspense>
            <Outlet />
          </Suspense>
        </DashboardLayout>
      ),
      children: [
        // { element: <IndexPage />, index: true },
        { path: 'user', element: <UserPage /> },
        { path: 'dashboard', element: <IndexPage /> },
        { path: 'submit', element: <SubmittionPage /> },
        { path: 'academic', element: <AcademicPage category="academic"/> },
        { path: 'finance', element: <FinancePage category="finance"/> },
        { path: 'equipment', element: <EquipmentPage category="equipment"/> },
        { path: 'studentinfo', element: <StudentPage /> },
        { path: 'inbox', element: <StudentBox /> },
        // { path: 'blog', element: <BlogPage /> },
      ],
    },
    {
      path: 'signup',
      element: <SignUpPage />,
    },
    {
      path: '/',
      element: <LoginPage />,
    },
    {
      path: '404',
      element: <Page404 />,
    },
    // {
    //   path: '*',
    //   element: <Navigate to="/404" replace />,
    // },
  ]);

  return routes;
}
