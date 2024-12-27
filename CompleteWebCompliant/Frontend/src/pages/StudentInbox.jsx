import { Helmet } from 'react-helmet-async';

import Inbox from 'src/sections/StudentIbox/Inbox';


export default function StdIbox() {
  return (
    <>
      <Helmet>
        <title> StudentInbox | Compliant UI </title>
      </Helmet>

      <Inbox/>
    </>
  );
}
