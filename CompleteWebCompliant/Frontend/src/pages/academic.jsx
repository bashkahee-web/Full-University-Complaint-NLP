import { Helmet } from 'react-helmet-async';

import { AcademicView } from 'src/sections/academic';



// ----------------------------------------------------------------------

export default function AppPage() {
  return (
    <>
      <Helmet>
        <title> Academic | Minimal UI </title>
      </Helmet>

      <AcademicView />
    </>
  );
}
