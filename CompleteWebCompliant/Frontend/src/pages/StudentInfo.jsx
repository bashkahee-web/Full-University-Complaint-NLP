import { Helmet } from 'react-helmet-async';

import StudentData from 'src/sections/StudentComplaints/StudentData';




// ----------------------------------------------------------------------

export default function StudentPage() {
  return (
    <>
      <Helmet>
        <title> StudentInfo | Compliant UI </title>
      </Helmet>

      <StudentData/>
    </>
  );
}
